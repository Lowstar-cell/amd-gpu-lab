import argparse
import tempfile
import os
import subprocess
import sys
import numpy as np
import torch

from precompute_embeddings import build_embeddings
from scheduler import load_embeddings, NearestNeighborScheduler, embed_sequence
from dataset_utils import AMDLogDataset, collate_fn
from model import Encoder
from torch.utils.data import DataLoader


def run_demo(data_dir, out_emb_path, model_ckpt=None, n_query=3):
    # Build embeddings
    build_embeddings(data_dir, out_emb_path, model_ckpt=model_ckpt)
    # Load embeddings
    embeddings, metadata = load_embeddings(out_emb_path)
    scheduler = NearestNeighborScheduler(embeddings, metadata)

    dataset = AMDLogDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    encoder = Encoder(len(dataset.token2id))

    # Query on few random samples
    for i, batch in enumerate(loader):
        if i >= n_query:
            break
        x, label, lengths = batch
        z = embed_sequence(encoder, x, lengths, device='cpu')
        res = scheduler.query(z[0], k=3)
        print(f"Query {i}: label={int(label.item())}, schedule={res['device']}, neighbors={len(res['neighbors'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out', default='./embeddings/embeddings.npz')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()
    run_demo(args.data_dir, args.out, n_query=args.n)
