"""
GPU Embedding Service - Selbstlernendes System

Läuft als Daemon, hält Model permanent auf GPU, lernt kontinuierlich.
"""

import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import mmap
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Füge Parent-Verzeichnis hinzu für imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Encoder


@dataclass
class WorkloadSample:
    """Ein Workload mit Features und gemessener Performance"""
    features: np.ndarray          # Input features
    embedding: Optional[np.ndarray] = None  # Computed embedding
    strategy_used: str = "default"  # Welche Strategie wurde angewendet
    performance_ms: float = 0.0   # Gemessene Ausführungszeit
    baseline_ms: float = 0.0      # Erwartete/CUDA Baseline Zeit
    reward: float = 0.0           # performance_ms / baseline_ms (kleiner = besser)
    timestamp: float = field(default_factory=time.time)


class ReplayBuffer:
    """Experience Replay für Online Learning"""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def add(self, sample: WorkloadSample):
        with self.lock:
            self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[WorkloadSample]:
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class StrategyNetwork(nn.Module):
    """
    Netzwerk das basierend auf Embedding die beste Strategie wählt.
    Output: Wahrscheinlichkeiten für verschiedene Optimierungsstrategien
    """

    STRATEGIES = [
        "default",           # Keine spezielle Optimierung
        "memory_coalesce",   # Memory access pattern optimieren
        "wavefront_tune",    # AMD Wavefront (64) vs CUDA Warp (32)
        "occupancy_boost",   # Mehr concurrent threads
        "cache_optimize",    # L2 cache besser nutzen
        "async_copy",        # Async memory transfers
    ]

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.STRATEGIES))
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns logits for each strategy"""
        return self.net(embedding)

    def get_strategy(self, embedding: torch.Tensor, explore: bool = True) -> Tuple[str, int]:
        """Wählt Strategie (mit optionaler Exploration)"""
        with torch.no_grad():
            logits = self.forward(embedding)
            probs = F.softmax(logits, dim=-1)

            if explore and np.random.random() < 0.1:  # 10% exploration
                action = np.random.randint(0, len(self.STRATEGIES))
            else:
                action = probs.argmax().item()

            return self.STRATEGIES[action], action


class GPUEmbeddingService:
    """
    Hauptservice - läuft dauerhaft, hält Model auf GPU, lernt kontinuierlich.

    Features:
    - Model permanent auf GPU (keine Startup-Latenz)
    - Automatisches Batching
    - Online Learning mit Reward-Signal
    - Shared Memory IPC für Clients
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        batch_timeout_ms: float = 5.0,
        max_batch_size: int = 64,
        learning_rate: float = 1e-4,
        learn_interval: int = 100,  # Alle N samples lernen
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[GPUService] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[GPUService] GPU: {torch.cuda.get_device_name(0)}")

        # Models
        self.encoder = Encoder(vocab_size=1000, embed_dim=64, hidden_dim=128, projection_dim=64)
        self.strategy_net = StrategyNetwork(embedding_dim=64)

        # Load checkpoint if exists
        if model_path and os.path.exists(model_path):
            self._load_checkpoint(model_path)

        # Move to GPU and keep there
        self.encoder.to(self.device)
        self.strategy_net.to(self.device)
        self.encoder.eval()

        # Batching
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.request_queue = queue.Queue()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.request_id = 0
        self.request_lock = threading.Lock()

        # Learning
        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.learn_interval = learn_interval
        self.samples_since_learn = 0
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.strategy_net.parameters()),
            lr=learning_rate
        )

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
            "learn_steps": 0,
            "avg_reward": 0.0,
        }

        # Background threads
        self._running = False
        self._batch_thread = None
        self._learn_thread = None

    def _load_checkpoint(self, path: str):
        """Load model checkpoint"""
        try:
            ckpt = torch.load(path, map_location=self.device)
            if 'encoder_state_dict' in ckpt:
                self.encoder.load_state_dict(ckpt['encoder_state_dict'])
            if 'strategy_state_dict' in ckpt:
                self.strategy_net.load_state_dict(ckpt['strategy_state_dict'])
            print(f"[GPUService] Loaded checkpoint: {path}")
        except Exception as e:
            print(f"[GPUService] Could not load checkpoint: {e}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'strategy_state_dict': self.strategy_net.state_dict(),
            'stats': self.stats,
        }, path)
        print(f"[GPUService] Saved checkpoint: {path}")

    def start(self):
        """Start background processing threads"""
        self._running = True
        self._batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self._learn_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._batch_thread.start()
        self._learn_thread.start()
        print("[GPUService] Started")

    def stop(self):
        """Stop background threads"""
        self._running = False
        if self._batch_thread:
            self._batch_thread.join(timeout=2.0)
        if self._learn_thread:
            self._learn_thread.join(timeout=2.0)
        print("[GPUService] Stopped")

    def _batch_processor(self):
        """Background thread that batches requests and processes on GPU"""
        while self._running:
            batch = []
            batch_ids = []
            deadline = time.time() + (self.batch_timeout_ms / 1000.0)

            # Collect batch
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    req_id, features = self.request_queue.get(timeout=0.001)
                    batch.append(features)
                    batch_ids.append(req_id)
                except queue.Empty:
                    continue

            if not batch:
                continue

            # Process batch on GPU
            start = time.time()
            try:
                embeddings, strategies = self._process_batch(batch)
                latency = (time.time() - start) * 1000

                # Update stats
                self.stats["total_batches"] += 1
                self.stats["total_requests"] += len(batch)
                alpha = 0.01
                self.stats["avg_batch_size"] = (1-alpha) * self.stats["avg_batch_size"] + alpha * len(batch)
                self.stats["avg_latency_ms"] = (1-alpha) * self.stats["avg_latency_ms"] + alpha * latency

                # Return results
                for i, req_id in enumerate(batch_ids):
                    if req_id in self.result_queues:
                        self.result_queues[req_id].put((
                            embeddings[i],
                            strategies[i]
                        ))
            except Exception as e:
                print(f"[GPUService] Batch error: {e}")
                for req_id in batch_ids:
                    if req_id in self.result_queues:
                        self.result_queues[req_id].put((None, "default"))

    def _process_batch(self, features_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[str]]:
        """Process a batch of features on GPU"""
        # Stack features
        # Für variable Längen: padding
        max_len = max(f.shape[0] for f in features_list)
        batch_size = len(features_list)

        # Assuming features are (seq_len, feat_dim) or flat
        if features_list[0].ndim == 1:
            # Flat features - reshape to sequence
            feat_dim = features_list[0].shape[0]
            padded = np.zeros((batch_size, max_len), dtype=np.float32)
            lengths = []
            for i, f in enumerate(features_list):
                padded[i, :len(f)] = f
                lengths.append(len(f))
        else:
            # Sequence features
            feat_dim = features_list[0].shape[1] if features_list[0].ndim > 1 else 1
            padded = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
            lengths = []
            for i, f in enumerate(features_list):
                padded[i, :f.shape[0]] = f
                lengths.append(f.shape[0])

        # To GPU
        x = torch.from_numpy(padded).long().to(self.device)
        lens = torch.tensor(lengths, device=self.device)

        # Forward pass
        with torch.no_grad():
            embeddings = self.encoder(x, lens)  # (batch, embed_dim)
            strategies = []
            for emb in embeddings:
                strategy, _ = self.strategy_net.get_strategy(emb.unsqueeze(0))
                strategies.append(strategy)

        return embeddings.cpu().numpy(), strategies

    def _learning_loop(self):
        """Background thread for continuous learning"""
        while self._running:
            time.sleep(1.0)  # Check every second

            if len(self.replay_buffer) < 32:
                continue

            if self.samples_since_learn >= self.learn_interval:
                self._learn_step()
                self.samples_since_learn = 0

    def _learn_step(self):
        """One learning step from replay buffer"""
        samples = self.replay_buffer.sample(32)
        if not samples:
            return

        self.encoder.train()
        self.strategy_net.train()

        # Prepare batch
        features = [s.features for s in samples]
        rewards = torch.tensor([s.reward for s in samples], device=self.device)
        actions = torch.tensor(
            [StrategyNetwork.STRATEGIES.index(s.strategy_used) for s in samples],
            device=self.device
        )

        # Forward
        embeddings, _ = self._process_batch_train(features)
        logits = self.strategy_net(embeddings)

        # Policy gradient loss
        # reward > 1 means worse than baseline, reward < 1 means better
        # We want to maximize (1 - reward) for chosen actions
        advantage = 1.0 - rewards  # positive = good
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(advantage * selected_log_probs).mean()

        # Contrastive loss for embeddings (optional, TODO)

        # Backward
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.strategy_net.parameters()),
            1.0
        )
        self.optimizer.step()

        self.encoder.eval()

        self.stats["learn_steps"] += 1
        self.stats["avg_reward"] = rewards.mean().item()

    def _process_batch_train(self, features_list: List[np.ndarray]) -> Tuple[torch.Tensor, List[str]]:
        """Process batch with gradients for training"""
        max_len = max(f.shape[0] for f in features_list)
        batch_size = len(features_list)

        if features_list[0].ndim == 1:
            padded = np.zeros((batch_size, max_len), dtype=np.float32)
            lengths = []
            for i, f in enumerate(features_list):
                padded[i, :len(f)] = f
                lengths.append(len(f))
        else:
            feat_dim = features_list[0].shape[1]
            padded = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
            lengths = []
            for i, f in enumerate(features_list):
                padded[i, :f.shape[0]] = f
                lengths.append(f.shape[0])

        x = torch.from_numpy(padded).long().to(self.device)
        lens = torch.tensor(lengths, device=self.device)
        embeddings = self.encoder(x, lens)

        return embeddings, []

    # === Public API ===

    def get_embedding_and_strategy(
        self,
        features: np.ndarray,
        timeout_ms: float = 100.0
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Get embedding and recommended strategy for a workload.

        Args:
            features: Workload features as numpy array
            timeout_ms: Max wait time

        Returns:
            (embedding, strategy_name) or (None, "default") on timeout
        """
        with self.request_lock:
            req_id = self.request_id
            self.request_id += 1
            self.result_queues[req_id] = queue.Queue()

        self.request_queue.put((req_id, features))

        try:
            result = self.result_queues[req_id].get(timeout=timeout_ms / 1000.0)
            return result
        except queue.Empty:
            return None, "default"
        finally:
            del self.result_queues[req_id]

    def report_performance(
        self,
        features: np.ndarray,
        embedding: np.ndarray,
        strategy_used: str,
        performance_ms: float,
        baseline_ms: float
    ):
        """
        Report actual performance for learning.

        Args:
            features: Original input features
            embedding: Computed embedding
            strategy_used: Which strategy was applied
            performance_ms: Actual measured time
            baseline_ms: Expected/baseline time (e.g., CUDA performance)
        """
        reward = performance_ms / max(baseline_ms, 1e-6)  # <1 = better than baseline

        sample = WorkloadSample(
            features=features,
            embedding=embedding,
            strategy_used=strategy_used,
            performance_ms=performance_ms,
            baseline_ms=baseline_ms,
            reward=reward
        )

        self.replay_buffer.add(sample)
        self.samples_since_learn += 1

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            **self.stats,
            "buffer_size": len(self.replay_buffer),
            "device": str(self.device),
        }


# === Singleton für einfachen Zugriff ===
_service_instance: Optional[GPUEmbeddingService] = None


def get_service() -> GPUEmbeddingService:
    """Get or create the global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = GPUEmbeddingService()
        _service_instance.start()
    return _service_instance


def shutdown_service():
    """Shutdown the global service"""
    global _service_instance
    if _service_instance is not None:
        _service_instance.stop()
        _service_instance = None


if __name__ == "__main__":
    # Test
    print("=== GPU Embedding Service Test ===")

    service = GPUEmbeddingService()
    service.start()

    # Simulate workloads
    for i in range(10):
        features = np.random.randint(0, 100, size=(32,)).astype(np.float32)
        embedding, strategy = service.get_embedding_and_strategy(features)
        print(f"[{i}] Strategy: {strategy}, Embedding shape: {embedding.shape if embedding is not None else None}")

        # Simulate performance feedback
        if embedding is not None:
            perf_ms = np.random.uniform(5, 20)
            baseline_ms = 10.0
            service.report_performance(features, embedding, strategy, perf_ms, baseline_ms)

    time.sleep(2)  # Let learning happen

    print(f"\nStats: {service.get_stats()}")
    service.stop()
