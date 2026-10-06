#!/usr/bin/env python3
"""
Train CUDA → HIP translator model.

Usage:
    python train_translator.py --epochs 100 --batch-size 16
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from translator_model import Seq2SeqTranslator, CharTokenizer


class CudaHipDataset(Dataset):
    """Dataset of CUDA → HIP code pairs."""

    def __init__(self, data_path, tokenizer, max_len=200):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cuda_code = item['cuda']
        hip_code = item['hip']

        # Tokenize
        src_tokens = self.tokenizer.encode(cuda_code, add_sos_eos=False)
        tgt_tokens = self.tokenizer.encode(hip_code, add_sos_eos=True)

        # Truncate if needed
        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]

        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_text': cuda_code,
            'tgt_text': hip_code
        }


def collate_fn(batch):
    """Collate batch with padding."""
    # Find max lengths
    max_src_len = max(item['src'].size(0) for item in batch)
    max_tgt_len = max(item['tgt'].size(0) for item in batch)

    # Pad sequences
    src_padded = []
    tgt_padded = []

    for item in batch:
        src = item['src']
        tgt = item['tgt']

        # Pad
        src_pad = torch.cat([src, torch.zeros(max_src_len - src.size(0), dtype=torch.long)])
        tgt_pad = torch.cat([tgt, torch.zeros(max_tgt_len - tgt.size(0), dtype=torch.long)])

        src_padded.append(src_pad)
        tgt_padded.append(tgt_pad)

    return {
        'src': torch.stack(src_padded),
        'tgt': torch.stack(tgt_padded)
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc='Training'):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(src, tgt, teacher_forcing_ratio=0.5)

        # Compute loss (ignore padding)
        # logits: (batch, tgt_len-1, vocab_size)
        # tgt: (batch, tgt_len)
        # We predict tgt[1:] from tgt[:-1]
        targets = tgt[:, 1:logits.size(1)+1].contiguous()
        logits = logits.contiguous()

        # Flatten
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Stats
        total_loss += loss.item() * targets.numel()
        total_tokens += (targets != 0).sum().item()  # Don't count padding

    return total_loss / max(total_tokens, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model - simplified version using only loss."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # Forward (with teacher forcing for loss computation)
            try:
                logits = model(src, tgt, teacher_forcing_ratio=1.0)

                # Compute loss
                targets = tgt[:, 1:logits.size(1)+1].contiguous()
                logits = logits.contiguous()

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                total_loss += loss.item() * src.size(0)
                total_samples += src.size(0)
            except Exception as e:
                # Skip problematic batches
                print(f"Skipping batch due to error: {e}")
                continue

    avg_loss = total_loss / max(total_samples, 1)
    # Return dummy accuracy for now
    return avg_loss, 0.0


def test_translation(model, tokenizer, cuda_code, device):
    """Test translation on a single example."""
    model.eval()
    with torch.no_grad():
        # Encode
        src_tokens = tokenizer.encode(cuda_code, add_sos_eos=False)
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)

        # Translate
        output_tokens = model.translate(src, max_len=200)

        # Decode
        hip_code = tokenizer.decode(output_tokens[0].cpu().tolist())

    return hip_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='cuda_translator/training_data.json')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model-out', default='cuda_translator/translator.pt')
    parser.add_argument('--eval-interval', type=int, default=10)

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    with open(args.data, 'r') as f:
        data = json.load(f)

    # Build tokenizer
    tokenizer = CharTokenizer()
    all_texts = []
    for item in data:
        all_texts.append(item['cuda'])
        all_texts.append(item['hip'])
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    # Simple split: 80% train, 20% val
    split_idx = int(len(data) * 0.8)

    # Save train/val split
    train_data_path = Path(args.data).parent / 'train_data.json'
    val_data_path = Path(args.data).parent / 'val_data.json'

    with open(train_data_path, 'w') as f:
        json.dump(data[:split_idx], f, indent=2)
    with open(val_data_path, 'w') as f:
        json.dump(data[split_idx:], f, indent=2)

    train_dataset = CudaHipDataset(train_data_path, tokenizer)
    val_dataset = CudaHipDataset(val_data_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model (dropout=0 wegen ROCm/MIOpen Bug)
    model = Seq2SeqTranslator(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"  val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Test on example
            test_cuda = "cudaMalloc(&ptr, size);"
            test_hip = test_translation(model, tokenizer, test_cuda, device)
            print(f"  Test: '{test_cuda}' → '{test_hip}'")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'tokenizer_char2id': tokenizer.char2id,
                    'tokenizer_id2char': tokenizer.id2char,
                    'vocab_size': tokenizer.vocab_size,
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, args.model_out)
                print(f"  Saved best model to {args.model_out}")

    print("Training complete!")


if __name__ == '__main__':
    main()
