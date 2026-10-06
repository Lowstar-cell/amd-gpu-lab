#!/usr/bin/env python3
"""Simple inference helper for saved PyTorch checkpoints.

Goal: make it easy to run a trained model on Windows (CPU) for a single sample.

Supported checkpoints:
- classifier checkpoints saved by `train.py` with keys: model_state_dict, vocab
- encoder/projection checkpoints saved by `train_contrastive.py` with keys: encoder_state_dict, proj_state_dict, vocab

Usage examples:
  python scripts/infer_model.py --model test_model.pt --sample 1,2,3,4
  python scripts/infer_model.py --model logs/checkpoint_latest.pt --sample-file sample_ids.txt

This is intentionally light-weight: it maps common checkpoint layouts to the model classes in model.py
and runs a single forward pass on CPU.
"""

import argparse
import json
import os
import sys
from typing import List

import torch

# ensure repo root is on path so we can import model.py
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model import LSTMClassifier, Encoder


def load_ckpt(path: str):
    ck = torch.load(path, map_location="cpu")
    return ck


def fix_state_dict(sd: dict):
    # Remove module. prefix if present
    new = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new


def parse_sample(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [int(p) for p in parts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to a .pt checkpoint")
    p.add_argument("--sample", help="Comma separated integer token ids, e.g. '1,2,3,4'")
    p.add_argument("--sample-file", help="File with comma-separated ids or json array")
    p.add_argument("--vocab-size", type=int, default=None, help="Fallback vocab size if checkpoint doesn't include vocab")
    p.add_argument("--type", choices=['auto', 'classifier', 'encoder'], default='auto', help='Force model type')
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(2)

    ck = load_ckpt(args.model)

    # Determine model type
    keys = set(ck.keys()) if isinstance(ck, dict) else set()
    model_type = args.type
    if model_type == 'auto':
        if 'model_state_dict' in keys or 'model_state' in keys:
            model_type = 'classifier'
        elif 'encoder_state_dict' in keys or 'encoder_state' in keys or ('encoder_state_dict' in keys and 'proj_state_dict' in keys):
            model_type = 'encoder'
        else:
            # Might be a raw state dict (no wrapping)
            # Inspect keys for typical classifier params (embedding.weight)
            if any(k.startswith('embedding.') or 'embedding.weight' in k for k in keys):
                model_type = 'classifier'
            else:
                model_type = 'classifier'  # default

    print(f"Detected model type: {model_type}")

    vocab = None
    if isinstance(ck, dict) and 'vocab' in ck:
        try:
            vocab = ck['vocab']
            if isinstance(vocab, dict):
                vocab_size = len(vocab)
            else:
                vocab_size = int(vocab)
        except Exception:
            vocab = None
            vocab_size = args.vocab_size or 1000
    else:
        vocab_size = args.vocab_size or 1000

    print(f"Vocab size (inferred/fallback): {vocab_size}")

    device = torch.device('cpu')

    if model_type == 'classifier':
        m = LSTMClassifier(vocab_size=vocab_size)
        # get state dict
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            sd = ck['model_state_dict']
        elif isinstance(ck, dict) and 'model_state' in ck:
            sd = ck['model_state']
        elif isinstance(ck, dict) and any(k.startswith('embedding.') for k in ck.keys()):
            sd = ck
        else:
            print('Could not find a model state dict in checkpoint. Trying to treat file as raw state dict...')
            sd = ck

        sd = fix_state_dict(sd)
        try:
            m.load_state_dict(sd)
        except Exception as e:
            print('Failed to load state_dict directly:', e)
            # try partial load
            m_state = m.state_dict()
            new_state = {}
            for k in m_state.keys():
                if k in sd:
                    new_state[k] = sd[k]
            m_state.update(new_state)
            m.load_state_dict(m_state)

        m.eval()

    else:  # encoder
        m = Encoder(vocab_size=vocab_size)
        if isinstance(ck, dict) and 'encoder_state_dict' in ck:
            sd = ck['encoder_state_dict']
        elif isinstance(ck, dict) and any(k.startswith('embedding.') for k in ck.keys()):
            sd = ck
        else:
            print('Could not find encoder_state_dict; using checkpoint as-is.')
            sd = ck

        sd = fix_state_dict(sd)
        try:
            m.load_state_dict(sd)
        except Exception as e:
            print('Failed to load encoder state_dict directly:', e)
            # try partial
            m_state = m.state_dict()
            new_state = {}
            for k in m_state.keys():
                if k in sd:
                    new_state[k] = sd[k]
            m_state.update(new_state)
            m.load_state_dict(m_state)

        m.eval()

    # Build sample tensor
    sample_ids = None
    if args.sample:
        sample_ids = parse_sample(args.sample)
    elif args.sample_file:
        if not os.path.exists(args.sample_file):
            print(f"Sample file not found: {args.sample_file}")
            sys.exit(3)
        txt = open(args.sample_file, 'r', encoding='utf-8').read().strip()
        # allow JSON array or comma separated
        try:
            j = json.loads(txt)
            if isinstance(j, list):
                sample_ids = [int(x) for x in j]
        except Exception:
            sample_ids = parse_sample(txt)

    if sample_ids is None:
        print('No sample tokens provided. Providing a dummy sample [1,2,3,4] for demonstration.')
        sample_ids = [1, 2, 3, 4]

    # unsqueeze batch axis
    import torch as _t
    x = _t.tensor([sample_ids], dtype=_t.long, device=device)

    with torch.no_grad():
        out = m(x)

    # Print result
    if isinstance(out, torch.Tensor):
        print('Output tensor:', out.cpu().numpy())
    else:
        print('Output:', out)


if __name__ == '__main__':
    main()
