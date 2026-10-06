"""
GPU Rasterization Data Collector
Sammelt Features während der Rasterisierung für Training
"""

import numpy as np
import torch
from typing import Dict, List
import h5py
import os

class RasterizationDataCollector:
    """
    Sammelt Daten während GPU Rasterization:
    - Barycentrische Koordinaten
    - Depth values
    - Coverage maps
    - Fragment frequencies
    """

    def __init__(self, save_dir: str = './data'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # allow configurable feature sets
        # default behavior preserves compatibility with existing pipeline (7-dims)
        self.features = []
        self.labels = []
        # collection options (tweakable by a higher-level CLI)
        self.options = {
            'feature_set': 'basic',  # basic or extended
            'compute_gradients': False,
            'compute_density': False,
            'compute_edges': False,
        }

    def collect_from_rasterization(
        self,
        barycentric_coords: np.ndarray,  # (H, W, N_triangles, 3)
        depth: np.ndarray,                # (H, W)
        coverage: np.ndarray,             # (H, W)
        triangle_ids: np.ndarray = None   # (H, W) - welches Triangle sichtbar
    ):
        """
        Sammle Features von einem Rasterization Pass
        """
        H, W = depth.shape
        num_triangles = barycentric_coords.shape[2]

        # Für jedes gerenderte Pixel
        covered_pixels = np.where(coverage)

        for y, x in zip(*covered_pixels):
            # Finde welches Triangle sichtbar ist (niedrigster Z)
            visible_tri = None
            min_z = float('inf')

            for t in range(num_triangles):
                w = barycentric_coords[y, x, t]
                if w.sum() > 0.99 and w.min() >= 0:  # Valid barycentric
                    if depth[y, x] < min_z:
                        min_z = depth[y, x]
                        visible_tri = t

            if visible_tri is not None:
                # Feature vector base (7-dim) — compatible with previous behavior
                w = barycentric_coords[y, x, visible_tri]
                base = [w[0], w[1], w[2], depth[y, x], float(x) / W, float(y) / H, 1.0]

                # Extended features
                extras = []
                if self.options.get('compute_gradients'):
                    # depth gradients at this pixel
                    # the caller may compute advanced features and attach them in attrs
                    # here we attempt to access precomputed arrays in the runtime if available
                    # default to 0 if not present
                    grad_dx = getattr(self, '_last_grad_dx', None)
                    grad_dy = getattr(self, '_last_grad_dy', None)
                    if grad_dx is not None and grad_dy is not None:
                        extras.append(float(grad_dx[y, x]))
                        extras.append(float(grad_dy[y, x]))
                    else:
                        extras.extend([0.0, 0.0])

                if self.options.get('compute_density'):
                    dens = getattr(self, '_last_density', None)
                    extras.append(float(dens[y, x]) if dens is not None else 0.0)

                if self.options.get('compute_edges'):
                    edges = getattr(self, '_last_edges', None)
                    extras.append(float(edges[y, x]) if edges is not None else 0.0)

                feature = np.array(list(base) + extras)

                self.features.append(feature)
                self.labels.append(visible_tri)

        print(f"[DataCollector] Collected {len(covered_pixels[0])} samples")

    def collect_batch(
        self,
        rasterizer_output: Dict,
        batch_idx: int = 0
    ):
        """Sammle von einem kompletten Batch"""
        if 'barycentric_coords' in rasterizer_output:
            self.collect_from_rasterization(
                rasterizer_output['barycentric_coords'],
                rasterizer_output['depth'],
                rasterizer_output['coverage']
            )

    def save_dataset(self, filename: str = 'rasterization_data.h5'):
        """Speichere gesammelte Daten als HDF5"""
        if len(self.features) == 0:
            print(f"[DataCollector] WARNING: No samples to save!")
            return None

        filepath = os.path.join(self.save_dir, filename)

        features_array = np.array(self.features, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)


        # Persist feature names depending on what we actually collected
        feature_names = None
        base_names = ['w0', 'w1', 'w2', 'depth', 'x_norm', 'y_norm', 'coverage']

        extras = []
        if self.options.get('compute_gradients'):
            extras += ['depth_dx', 'depth_dy']
        if self.options.get('compute_density'):
            extras += ['fragment_density']
        if self.options.get('compute_edges'):
            extras += ['edge_strength']

        expected_dim = len(base_names) + len(extras)
        if features_array.shape[1] == expected_dim:
            feature_names = base_names + extras

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('features', data=features_array, compression='gzip')
            f.create_dataset('labels', data=labels_array, compression='gzip')
            if feature_names is not None:
                # persist as an attribute so other tools can reconstruct column names
                try:
                    # store as a single string attribute for portability
                    f.attrs['feature_names'] = ','.join(feature_names)
                except Exception:
                    pass

            # Metadata
            f.attrs['num_samples'] = len(self.features)
            f.attrs['feature_dim'] = features_array.shape[1] if len(features_array.shape) > 1 else 0
            f.attrs['description'] = 'GPU Rasterization Training Data'

        print(f"[DataCollector] Saved {len(self.features)} samples to {filepath}")
        return filepath

    def load_dataset(self, filename: str = 'rasterization_data.h5'):
        """Lade gespeicherte Daten"""
        filepath = os.path.join(self.save_dir, filename)

        with h5py.File(filepath, 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]

        return features, labels

    def get_pytorch_dataset(self):
        """Konvertiere zu PyTorch Dataset"""
        features = torch.tensor(np.array(self.features), dtype=torch.float32)
        labels = torch.tensor(np.array(self.labels), dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(features, labels)
        return dataset

    def clear(self):
        """Clear collected data"""
        self.features = []
        self.labels = []

    def compute_and_attach_derived(self, depth: np.ndarray, coverage: np.ndarray, kernel_size: int = 5):
        """Compute derived feature arrays and attach to collector for per-pixel lookup.

        This allows collect_from_rasterization to include per-pixel gradient/density/edge features
        if the options are enabled.
        """
        # gradients
        try:
            dy, dx = np.gradient(depth)
            self._last_grad_dx = dx
            self._last_grad_dy = dy
        except Exception:
            self._last_grad_dx = None
            self._last_grad_dy = None

        # density (local fragment density)
        if self.options.get('compute_density'):
            try:
                from scipy.ndimage import uniform_filter
                self._last_density = uniform_filter(coverage.astype(float), size=kernel_size)
            except Exception:
                self._last_density = None

        # edges
        if self.options.get('compute_edges'):
            try:
                from scipy.ndimage import sobel
                edges_x = sobel(coverage.astype(float), axis=1)
                edges_y = sobel(coverage.astype(float), axis=0)
                self._last_edges = np.hypot(edges_x, edges_y)
            except Exception:
                self._last_edges = None


class RasterizationDataAugmenter:
    """
    Data Augmentation für Rasterization Features
    """

    @staticmethod
    def rotate_barycentric(barycentric: np.ndarray, angle: float) -> np.ndarray:
        """Rotiere Triangle (durch Permutation der barycentric coords)"""
        # Zyklische Rotation der barycentric coordinates
        if angle == 120:
            return np.array([barycentric[1], barycentric[2], barycentric[0]])
        elif angle == 240:
            return np.array([barycentric[2], barycentric[0], barycentric[1]])
        return barycentric

    @staticmethod
    def flip_barycentric(barycentric: np.ndarray, axis: str = 'x') -> np.ndarray:
        """Spiegel Triangle"""
        if axis == 'x':
            return np.array([barycentric[1], barycentric[0], barycentric[2]])
        elif axis == 'y':
            return np.array([barycentric[0], barycentric[2], barycentric[1]])
        return barycentric

    @staticmethod
    def add_noise(features: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """Füge Rauschen zu Features hinzu"""
        noise = np.random.randn(*features.shape) * noise_scale
        return features + noise


# Feature Extractor für fortgeschrittene Datensammlung
class AdvancedFeatureExtractor:
    """
    Extrahiert komplexe Features aus Rasterization
    """

    @staticmethod
    def compute_gradients(depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Berechne Depth-Gradienten"""
        dy, dx = np.gradient(depth)

        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        return {
            'depth_dx': dx,
            'depth_dy': dy,
            'gradient_magnitude': magnitude,
            'gradient_angle': angle
        }

    @staticmethod
    def compute_fragment_density(coverage: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Berechne lokale Fragment-Dichte
        (wie viele Pixel sind in der Nähe gerendert?)
        """
        from scipy.ndimage import uniform_filter

        density = uniform_filter(coverage.astype(float), size=kernel_size)
        return density

    @staticmethod
    def extract_edge_features(coverage: np.ndarray) -> Dict[str, np.ndarray]:
        """Extrahiere Kanten-Features (Triangle Boundaries)"""
        from scipy.ndimage import sobel

        edges_x = sobel(coverage.astype(float), axis=1)
        edges_y = sobel(coverage.astype(float), axis=0)
        edges = np.hypot(edges_x, edges_y)

        return {
            'edges': edges,
            'edges_x': edges_x,
            'edges_y': edges_y
        }


if __name__ == '__main__':
    # Test
    print("=== Data Collector Test ===")

    collector = RasterizationDataCollector(save_dir='./data')

    # Simuliere Rasterization Output
    H, W, N = 256, 256, 10

    fake_barycentric = np.random.rand(H, W, N, 3)
    fake_barycentric /= fake_barycentric.sum(axis=3, keepdims=True)

    fake_depth = np.random.rand(H, W) * 100
    fake_coverage = (np.random.rand(H, W) > 0.7).astype(bool)

    collector.collect_from_rasterization(
        fake_barycentric,
        fake_depth,
        fake_coverage
    )

    # Save
    collector.save_dataset('test_data.h5')

    # Load
    features, labels = collector.load_dataset('test_data.h5')
    print(f"Loaded: features {features.shape}, labels {labels.shape}")

    print("\n[SUCCESS] Data Collector funktioniert!")
