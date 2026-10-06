import argparse
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import AMDLogDataset, collate_fn
from model import Encoder, ProjectionHead


def random_token_mask(tokens: torch.Tensor, lengths: torch.Tensor, mask_token: int = 1, p_mask: float = 0.1):
    # tokens: (batch, seq_len), lengths: (batch,)
    b, seqlen = tokens.shape
    out = tokens.clone()
    for i in range(b):
        L = lengths[i]
        if L == 0:
            continue
        # choose positions to mask within valid length
        mask_positions = torch.rand(L) < p_mask
        out[i, :L][mask_positions] = mask_token
    return out


def subsample_sequence(tokens: torch.Tensor, lengths: torch.Tensor, min_len: int = 3):
    b, seqlen = tokens.shape
    out = torch.zeros_like(tokens)
    new_lengths = torch.zeros_like(lengths)
    for i in range(b):
        L = int(lengths[i].item())
        if L <= min_len:
            out[i, :L] = tokens[i, :L]
            new_lengths[i] = L
            continue
        newL = random.randint(min_len, L)
        # choose random start such that [start:start+newL] fits in L
        start = random.randint(0, L - newL)
        out[i, :newL] = tokens[i, start:start+newL]
        new_lengths[i] = newL
    return out, new_lengths


def token_drop(tokens: torch.Tensor, lengths: torch.Tensor, p_drop: float = 0.05):
    # Replace some tokens with pad (0) to simulate event loss/drop
    b, seqlen = tokens.shape
    out = tokens.clone()
    for i in range(b):
        L = int(lengths[i].item())
        if L == 0:
            continue
        mask_positions = torch.rand(L) < p_drop
        out[i, :L][mask_positions] = 0
    return out


def token_shuffle(tokens: torch.Tensor, lengths: torch.Tensor, max_permute: int = 3):
    # For each sequence, perform a few random swaps within the valid prefix to simulate reordering
    out = tokens.clone()
    b, seqlen = tokens.shape
    for i in range(b):
        L = int(lengths[i].item())
        if L <= 1:
            continue
        nperm = min(max_permute, max(1, L // 8))
        for _ in range(nperm):
            a = random.randint(0, L - 1)
            bpos = random.randint(0, L - 1)
            tmp = out[i, a].item()
            out[i, a] = out[i, bpos]
            out[i, bpos] = tmp
    return out


def augment_view(tokens: torch.Tensor, lengths: torch.Tensor, mask_p=0.15, drop_p=0.05):
    # Compose augmentation pipeline: subsample -> mask -> drop -> shuffle
    out, new_lengths = subsample_sequence(tokens, lengths)
    out = random_token_mask(out, new_lengths, p_mask=mask_p)
    out = token_drop(out, new_lengths, p_drop=drop_p)
    out = token_shuffle(out, new_lengths)
    return out, new_lengths


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    # z1, z2: (batch, dim)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    bs, d = z1.shape
    mat = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(bs, device=z1.device)
    loss1 = nn.CrossEntropyLoss()(mat, labels)
    loss2 = nn.CrossEntropyLoss()(mat.t(), labels)
    return (loss1 + loss2) / 2.0


def train_contrastive(encoder: Encoder, proj: ProjectionHead, loader: DataLoader, opt, epochs: int, device: torch.device, val_loader=None, scheduler=None, save_best=None, save_every=1, writer: SummaryWriter = None):
    encoder.train(); proj.train()
    encoder.to(device); proj.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for batch in tqdm(loader, desc=f"epoch {epoch+1}"):
            x, _, lengths = batch
            x = x.to(device)
            lengths = lengths.to(device)
            # two augmented views
            xa, la = augment_view(x, lengths, mask_p=0.15, drop_p=0.05)
            xb, lb = augment_view(x, lengths, mask_p=0.15, drop_p=0.05)
            xa = xa.to(device); la = la.to(device)
            xb = xb.to(device); lb = lb.to(device)

            za = proj(encoder(xa, la))
            zb = proj(encoder(xb, lb))

            loss = info_nce_loss(za, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        avg = total_loss / max(1, total)
        # validate
        val_loss = None
        if val_loader is not None:
            val_loss = 0.0
            val_n = 0
            encoder.eval(); proj.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x, _, lengths = batch
                    x = x.to(device); lengths = lengths.to(device)
                    xa, la = augment_view(x, lengths, mask_p=0.15, drop_p=0.05)
                    xb, lb = augment_view(x, lengths, mask_p=0.15, drop_p=0.05)
                    za = proj(encoder(xa, la))
                    zb = proj(encoder(xb, lb))
                    loss = info_nce_loss(za, zb)
                    val_loss += loss.item() * x.size(0)
                    val_n += x.size(0)
            val_loss = val_loss / max(1, val_n)
            encoder.train(); proj.train()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        print(f"Epoch {epoch+1}/{epochs} contrastive loss: {avg:.4f}", end='')
        if writer is not None:
            writer.add_scalar('contrastive/train_loss', avg, epoch+1)
            if val_loss is not None:
                writer.add_scalar('contrastive/val_loss', val_loss, epoch+1)
        if val_loss is not None:
            print(f" val_loss: {val_loss:.4f}")
        else:
            print("")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best:
                torch.save({'encoder_state_dict': encoder.state_dict(), 'proj_state_dict': proj.state_dict()}, save_best)
                print(f"Saved best encoder to {save_best}")
        if save_every and (epoch + 1) % save_every == 0 and save_best:
            # periodic save
            torch.save({'encoder_state_dict': encoder.state_dict(), 'proj_state_dict': proj.state_dict()}, save_best + f".epoch{epoch+1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save-out', default='./contrastive_encoder.pt')
    parser.add_argument('--logdir', default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = AMDLogDataset(args.data_dir)
    # split into train and val
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    encoder = Encoder(len(dataset.token2id))
    proj = ProjectionHead(in_dim=encoder.proj[-1].in_features if hasattr(encoder.proj[-1], 'in_features') else 64, out_dim=64)
    # optimizer with weight decay
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr, weight_decay=1e-5)
    # LR scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    save_best = args.save_out
    writer = None
    if args.logdir:
        writer = SummaryWriter(log_dir=args.logdir)
    train_contrastive(encoder, proj, loader, opt, args.epochs, device, val_loader=val_loader, scheduler=scheduler, save_best=save_best, writer=writer)
    # Save
    out_path = save_best
    torch.save({'encoder_state_dict': encoder.state_dict(), 'proj_state_dict': proj.state_dict(), 'vocab': dataset.token2id}, out_path)
    if writer is not None:
        # log a small set of embeddings to TensorBoard projector
        try:
            sample_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
            x, labels, lengths = next(iter(sample_loader))
            z = proj(encoder(x, lengths)).cpu().detach().numpy()
            writer.add_embedding(z, metadata=[str(int(l.item())) for l in labels], global_step=0, tag='contrastive/embeddings')
        except Exception:
            pass
        writer.close()
    print(f"Saved encoder to {out_path}")


if __name__ == '__main__':
    main()
