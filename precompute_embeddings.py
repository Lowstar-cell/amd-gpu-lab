import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset_utils import AMDLogDataset, collate_fn
from model import Encoder
from scheduler import save_embeddings


def build_embeddings(data_dir, out_path, model_ckpt=None, batch_size=64, device='cpu'):
    dataset = AMDLogDataset(data_dir)
    vocab_size = len(dataset.token2id)
    encoder = Encoder(vocab_size)
    if model_ckpt and os.path.exists(model_ckpt):
        ckpt = torch.load(model_ckpt, map_location='cpu')
        # Attempt to load encoder weights; if the ckpt saved only classifier, skip
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_embs = []
    metadata = []
    for batch in loader:
        x, labels, lengths = batch
        with torch.no_grad():
            z = encoder(x, lengths)
        arr = z.cpu().numpy()
        all_embs.append(arr)
        # metadata per sample
        for i in range(z.shape[0]):
            pref_device = 'cpu' if labels[i].item() == 1 else 'gpu'
            metadata.append({'label': int(labels[i].item()), 'pref_device': pref_device})
    all_embs = np.concatenate(all_embs, axis=0)
    save_embeddings(out_path, all_embs, metadata)
    print(f"Saved embeddings ({all_embs.shape}) and metadata to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out', default='./embeddings/embeddings.npz')
    parser.add_argument('--model-ckpt', default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    build_embeddings(args.data_dir, args.out, model_ckpt=args.model_ckpt, batch_size=args.batch_size, device=args.device)


if __name__ == '__main__':
    main()
