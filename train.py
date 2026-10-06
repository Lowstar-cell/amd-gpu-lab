#!/usr/bin/env python3
import os
import argparse
import math
import time
import random
import json
import signal
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    from torch import nn
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch is not installed. Install a CPU build with: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Or, install a ROCm compatible build if you want to use AMD GPUs (follow AMD/ROCm installation instructions)."
    )
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset_utils import AMDLogDataset, collate_fn
from model import LSTMClassifier


def get_device(preference: str):
    # preference: 'auto', 'cpu', 'gpu'
    if preference == 'cpu':
        return torch.device('cpu')
    elif preference == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise RuntimeError('GPU requested but not available')
    else:
        # auto
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


def train_one_epoch(model, dataloader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in tqdm(dataloader, desc='train'):
        x, y, lengths = batch
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        opt.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(1, total_samples)


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
    acc = correct / max(1, total_samples)
    return total_loss / max(1, total_samples), acc


def _safe_mkdir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model-out', default='./model.pt')
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # checkpointing / resume
    parser.add_argument('--checkpoint-dir', default=None,
                        help='Directory to store checkpoints. Defaults to <logdir>/checkpoints or ./checkpoints')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from, or "auto" to find latest in checkpoint-dir')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Save a periodic checkpoint every N epochs (0 = disable)')
    parser.add_argument('--keep-last', type=int, default=5,
                        help='Keep only this many recent checkpoints (older removed)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except Exception:
        pass

    device = get_device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        # Provide basic ROCm info if available
        try:
            print("torch.cuda.is_available():", torch.cuda.is_available())
            print("torch.version.hip:", getattr(torch.version, 'hip', None))
        except Exception:
            pass

    dataset = AMDLogDataset(args.data_dir)

    # split deterministically
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(dataset.token2id)
    model = LSTMClassifier(vocab_size).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # tensorboard
    writer = None
    if args.logdir:
        _safe_mkdir(Path(args.logdir))
        writer = SummaryWriter(log_dir=args.logdir)
        # log config
        writer.add_text('run/config', json.dumps(vars(args), indent=2))

    # checkpoint directory
    if args.checkpoint_dir is None:
        checkpoint_dir = Path(args.logdir) / 'checkpoints' if args.logdir else Path('./checkpoints')
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    _safe_mkdir(checkpoint_dir)

    def _checkpoint_path(epoch):
        return checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'

    def _latest_checkpoint():
        latest = checkpoint_dir / 'latest.pt'
        if latest.exists():
            return latest
        files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _save_checkpoint(state, path: Path):
        tmp = path.with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(path)
        # update latest symlink (best-effort)
        try:
            latest = checkpoint_dir / 'latest.pt'
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(path.name)
        except Exception:
            try:
                torch.save(state, checkpoint_dir / 'latest.pt')
            except Exception:
                pass

    def _prune_checkpoints(keep: int):
        if keep <= 0:
            return
        files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'), key=lambda p: p.stat().st_mtime)
        while len(files) > keep:
            try:
                files[0].unlink()
            except Exception:
                pass
            files = files[1:]

    # interrupt handler
    interrupted = {'flag': False}

    def _handle_signal(signum, frame):
        interrupted['flag'] = True
        print(f"Received signal {signum} — will attempt a checkpoint before exiting.")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    best_val_loss = float('inf')
    start_epoch = 0

    # resume logic
    if args.resume:
        resume_path = args.resume
        if resume_path == 'auto':
            latest = _latest_checkpoint()
            if latest is None:
                print('No checkpoint found to resume (auto). Continuing from scratch.')
            else:
                resume_path = str(latest)

        if resume_path and resume_path != 'auto':
            cp = Path(resume_path)
            if not cp.exists():
                print(f'Requested checkpoint to resume from not found: {resume_path}. Continuing from scratch.')
            else:
                print(f'Loading checkpoint {resume_path} ...')
                # robust load: prefer full-state load, allowlist numpy reconstruct functions
                def _robust_torch_load(p):
                    try:
                        return torch.load(str(p), map_location='cpu', weights_only=False)
                    except Exception:
                        # try allowlisting known numpy reconstruction helpers
                        try:
                            import numpy as _np
                            candidates = []
                            if hasattr(_np, '__all__'):
                                pass
                            # common locations for the reconstruct function
                            if hasattr(_np, '_core') and hasattr(_np._core, 'multiarray') and hasattr(_np._core.multiarray, '_reconstruct'):
                                candidates.append(_np._core.multiarray._reconstruct)
                            if hasattr(_np, '_multiarray_umath') and hasattr(_np._multiarray_umath, '_reconstruct'):
                                candidates.append(_np._multiarray_umath._reconstruct)
                            if hasattr(_np, 'multiarray') and hasattr(_np.multiarray, '_reconstruct'):
                                candidates.append(_np.multiarray._reconstruct)
                            if candidates:
                                try:
                                    torch.serialization.add_safe_globals(candidates)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # retry
                        return torch.load(str(p), map_location='cpu', weights_only=False)

                state = _robust_torch_load(cp)
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                if 'optimizer_state_dict' in state and state['optimizer_state_dict'] is not None:
                    try:
                        opt.load_state_dict(state['optimizer_state_dict'])
                    except Exception:
                        print('Could not fully load optimizer state dict. Continuing with fresh optimizer state.')
                if 'epoch' in state:
                    start_epoch = int(state['epoch'])
                    print(f'Resuming from epoch {start_epoch}')
                if 'best_val_loss' in state:
                    best_val_loss = float(state['best_val_loss'])
                # restore RNGs if available
                try:
                    import numpy as _np
                    if 'np_rng' in state and state['np_rng'] is not None:
                        _np.random.set_state(state['np_rng'])
                    if 'py_rng' in state and state['py_rng'] is not None:
                        random.setstate(state['py_rng'])
                    if 'torch_rng' in state and state['torch_rng'] is not None:
                        torch.set_rng_state(state['torch_rng'])
                except Exception:
                    pass

    # training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch + 1)
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/acc', val_acc, epoch + 1)
            writer.add_scalar('timing/epoch_seconds', time.time() - epoch_start_time, epoch + 1)

        # periodic checkpoint
        if args.checkpoint_interval > 0 and ((epoch + 1) % args.checkpoint_interval == 0):
            cp_path = _checkpoint_path(epoch + 1)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
                'np_rng': None,
                'py_rng': None,
                'torch_rng': None,
            }
            try:
                import numpy as _np
                state['np_rng'] = _np.random.get_state()
            except Exception:
                pass
            try:
                state['py_rng'] = random.getstate()
            except Exception:
                pass
            try:
                state['torch_rng'] = torch.get_rng_state()
            except Exception:
                pass
            _save_checkpoint(state, cp_path)
            _prune_checkpoints(args.keep_last)
            print(f'Wrote checkpoint {cp_path}')

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'vocab': dataset.token2id, 'best_val_loss': best_val_loss}, args.model_out)
            print(f"Saved best model to {args.model_out}")

        # if signaled, save final checkpoint and stop
        if interrupted['flag']:
            print('Interrupted — writing final checkpoint before exit...')
            cp_path = _checkpoint_path(epoch + 1)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
                'np_rng': None,
                'py_rng': None,
                'torch_rng': None,
            }
            try:
                import numpy as _np
                state['np_rng'] = _np.random.get_state()
            except Exception:
                pass
            try:
                state['py_rng'] = random.getstate()
            except Exception:
                pass
            try:
                state['torch_rng'] = torch.get_rng_state()
            except Exception:
                pass
            _save_checkpoint(state, cp_path)
            print(f'Final checkpoint written: {cp_path}')
            break

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
