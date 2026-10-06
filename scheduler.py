import json
import numpy as np
import os
from typing import List, Dict
import torch
from model import Encoder


def save_embeddings(path: str, embeddings: np.ndarray, metadata: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # embeddings: (N, d), metadata: list of dicts length N
    np.savez_compressed(path, embeddings=embeddings, metadata=json.dumps(metadata))


def load_embeddings(path: str):
    d = np.load(path, allow_pickle=True)
    embeddings = d['embeddings']
    metadata = json.loads(str(d['metadata']))
    return embeddings, metadata


class NearestNeighborScheduler:
    def __init__(self, embedding_db: np.ndarray, metadata: List[Dict]):
        self.embedding_db = embedding_db
        self.metadata = metadata

    def query(self, emb: np.ndarray, k: int = 1):
        # emb: (d,) or (1, d)
        if emb.ndim == 1:
            emb = emb[None, :]
        # compute euclidean distances
        diff = self.embedding_db - emb
        dists = (diff ** 2).sum(axis=-1)
        idxs = np.argsort(dists)[:k]
        neighbors = [self.metadata[i] for i in idxs]
        # For prototype: decide schedule by majority device in neighbors
        devices = [n.get('pref_device', 'gpu') for n in neighbors]
        # majority vote
        device = max(set(devices), key=lambda x: devices.count(x))
        return {
            'device': device,
            'neighbors': neighbors,
            'distances': dists[idxs].tolist()
        }


def embed_sequence(encoder: Encoder, x: torch.Tensor, lengths: torch.Tensor, device='cpu'):
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        x = x.to(device)
        lengths = lengths.to(device)
        z = encoder(x, lengths).cpu().numpy()
    return z
