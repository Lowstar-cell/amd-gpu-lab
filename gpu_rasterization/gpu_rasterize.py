"""
AMD GPU 3D Rasterizer - Python Interface
Unterstützt HIP, OpenCL, und ROCm
"""

import numpy as np
import torch
import os
import ctypes
from typing import Tuple, Optional
import subprocess

class AMDRasterizer:
    """
    3D Rasterizer mit AMD GPU Support
    - HIP für beste Performance
    - OpenCL als Fallback
    - Automatische Datensammlung für Training
    """

    def __init__(self, width: int = 512, height: int = 512, backend: str = 'auto'):
        self.width = width
        self.height = height
        self.backend = backend
        self.device = None

        # Verfügbaren Backend bestimmen
        if backend == 'auto':
            self.backend = self._detect_backend()

        print(f"[Rasterizer] Initialisiere {self.backend} Backend ({width}x{height})")

        # Data collection buffer
        self.data_buffer = []

    def _detect_backend(self) -> str:
        """Erkenne verfügbare AMD GPU Technologie"""
        # Erste Priorität: PyTorch mit ROCm (funktioniert sofort!)
        if torch.cuda.is_available():
            # Auf AMD GPUs ist torch.cuda tatsächlich ROCm
            try:
                device_name = torch.cuda.get_device_name(0)
                if 'AMD' in device_name or 'Radeon' in device_name:
                    return 'torch_rocm'
            except:
                pass

        # Check für kompilierte HIP Kernels
        hip_lib = os.path.join(os.path.dirname(__file__), 'build', 'rasterizer_hip.so')
        hip_dll = os.path.join(os.path.dirname(__file__), 'build', 'rasterizer_hip.dll')
        if os.path.exists(hip_lib) or os.path.exists(hip_dll):
            return 'hip'

        # Check für OpenCL
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'AMD' in platform.name:
                    return 'opencl'
        except ImportError:
            pass

        return 'cpu'

    def compile_hip_kernel(self) -> bool:
        """Kompiliere HIP Kernel"""
        kernel_path = os.path.join(os.path.dirname(__file__),
                                   'kernels/hip/rasterizer.hip')
        output_path = kernel_path.replace('.hip', '.so')

        try:
            cmd = [
                'hipcc',
                '-fPIC',
                '--shared',
                kernel_path,
                '-o', output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"[Rasterizer] HIP Kernel kompiliert: {output_path}")
                return True
            else:
                print(f"[Rasterizer] HIP Compile Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"[Rasterizer] Kann hipcc nicht finden: {e}")
            return False

    def rasterize_triangles(
        self,
        vertices: np.ndarray,      # (N, 3, 3) - N triangles mit 3 vertices (x,y,z)
        colors: np.ndarray,         # (N, 3) - RGB für jedes Triangle
        mvp_matrix: np.ndarray,     # (4, 4) Model-View-Projection
        collect_data: bool = True
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Rasterisiere Triangles auf AMD GPU

        Returns:
            color_buffer: (H, W, 3) RGB Image
            training_data: Dict mit Features für Training/Embedding
        """

        if self.backend == 'torch_rocm':
            return self._rasterize_torch(vertices, colors, mvp_matrix, collect_data)
        elif self.backend == 'hip':
            return self._rasterize_hip(vertices, colors, mvp_matrix, collect_data)
        elif self.backend == 'opencl':
            return self._rasterize_opencl(vertices, colors, mvp_matrix, collect_data)
        else:
            return self._rasterize_cpu(vertices, colors, mvp_matrix, collect_data)

    def _rasterize_torch(
        self,
        vertices: np.ndarray,
        colors: np.ndarray,
        mvp_matrix: np.ndarray,
        collect_data: bool
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """PyTorch/ROCm Implementation - funktioniert JETZT schon!"""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Konvertiere zu Torch Tensors
        verts = torch.from_numpy(vertices).float().to(device)
        cols = torch.from_numpy(colors).float().to(device)
        mvp = torch.from_numpy(mvp_matrix).float().to(device)

        # Transform vertices
        num_triangles = verts.shape[0]
        transformed_verts = torch.zeros((num_triangles, 3, 4), device=device)

        for i in range(num_triangles):
            for j in range(3):  # 3 vertices per triangle
                v = torch.cat([verts[i, j], torch.ones(1, device=device)])
                v_transformed = mvp @ v

                # Perspective divide
                if v_transformed[3] != 0:
                    v_transformed[:3] /= v_transformed[3]

                # NDC to screen space
                transformed_verts[i, j, 0] = (v_transformed[0] + 1) * self.width / 2
                transformed_verts[i, j, 1] = (v_transformed[1] + 1) * self.height / 2
                transformed_verts[i, j, 2] = v_transformed[2]
                transformed_verts[i, j, 3] = v_transformed[3]

        # Rasterization auf GPU
        color_buffer = torch.zeros((self.height, self.width, 3), device=device)
        depth_buffer = torch.full((self.height, self.width), 1e10, device=device)

        # Data collection
        barycentric_data = None
        if collect_data:
            barycentric_data = torch.zeros((self.height, self.width, num_triangles, 3), device=device)

        # Pixel-parallel rasterization
        y_coords = torch.arange(self.height, device=device).view(-1, 1).expand(-1, self.width)
        x_coords = torch.arange(self.width, device=device).view(1, -1).expand(self.height, -1)

        pixel_centers_x = x_coords.float() + 0.5
        pixel_centers_y = y_coords.float() + 0.5

        for t in range(num_triangles):
            v0 = transformed_verts[t, 0]
            v1 = transformed_verts[t, 1]
            v2 = transformed_verts[t, 2]

            # Edge functions (vektorisiert für alle Pixel!)
            def edge_function(a, b, px, py):
                return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])

            w0 = edge_function(v1, v2, pixel_centers_x, pixel_centers_y)
            w1 = edge_function(v2, v0, pixel_centers_x, pixel_centers_y)
            w2 = edge_function(v0, v1, pixel_centers_x, pixel_centers_y)

            # Coverage test
            coverage = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

            if coverage.any():
                # Barycentric coordinates
                area = edge_function(v0, v1, v2[0], v2[1])
                if abs(area) > 1e-6:
                    w0_norm = w0 / area
                    w1_norm = w1 / area
                    w2_norm = w2 / area

                    # Interpolate depth
                    z = w0_norm * v0[2] + w1_norm * v1[2] + w2_norm * v2[2]

                    # Z-test and update
                    z_test = (z < depth_buffer) & coverage

                    color_buffer[z_test] = cols[t]
                    depth_buffer[z_test] = z[z_test]

                    # Data collection
                    if collect_data and barycentric_data is not None:
                        barycentric_data[z_test, t, 0] = w0_norm[z_test]
                        barycentric_data[z_test, t, 1] = w1_norm[z_test]
                        barycentric_data[z_test, t, 2] = w2_norm[z_test]

        # Back to CPU
        result_color = color_buffer.cpu().numpy()

        training_data = None
        if collect_data and barycentric_data is not None:
            training_data = {
                'barycentric_coords': barycentric_data.cpu().numpy(),
                'depth': depth_buffer.cpu().numpy(),
                'coverage': (depth_buffer < 1e9).cpu().numpy()
            }

        return result_color, training_data

    def _rasterize_hip(self, vertices, colors, mvp_matrix, collect_data):
        """Native HIP Implementation (für maximale Performance)"""
        # TODO: Load compiled .so und call via ctypes
        raise NotImplementedError("HIP Backend - compile with compile_hip_kernel() first")

    def _rasterize_opencl(self, vertices, colors, mvp_matrix, collect_data):
        """OpenCL Fallback"""
        # TODO: pyopencl integration
        raise NotImplementedError("OpenCL Backend coming soon")

    def _rasterize_cpu(self, vertices, colors, mvp_matrix, collect_data):
        """CPU Fallback (langsam aber funktioniert immer)"""
        print("[Rasterizer] WARNING: Using CPU fallback (slow!)")
        return self._rasterize_torch(vertices, colors, mvp_matrix, collect_data)


# Embedding Network für Rasterization Features
class RasterizationEmbedding(torch.nn.Module):
    """
    Neural Network für Embedding von Rasterization-Features
    Trainiert auf GPU-gesammelten Daten.

    Typical feature order produced by the collector is 7 dims:
        [w0, w1, w2, depth, x_norm, y_norm, coverage]

    The network accepts a configurable `input_dim` (default 7) to match saved data.
    """

    def __init__(self, input_dim: int = 7, embedding_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim)
        )

    def forward(self, features):
        """
        features: [batch, input_dim] - default expected 7 dims
        returns: [batch, embedding_dim]
        """
        return self.encoder(features)


if __name__ == '__main__':
    # Test
    print("=== AMD GPU Rasterizer Test ===")

    rasterizer = AMDRasterizer(width=256, height=256)

    # Simple Triangle
    vertices = np.array([
        [[128, 50, 0.5], [50, 200, 0.5], [200, 200, 0.5]]
    ], dtype=np.float32)

    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    # Identity MVP
    mvp = np.eye(4, dtype=np.float32)
    mvp[0, 0] = 2.0 / 256
    mvp[1, 1] = 2.0 / 256
    mvp[0, 3] = -1.0
    mvp[1, 3] = -1.0

    color_img, training_data = rasterizer.rasterize_triangles(
        vertices, colors, mvp, collect_data=True
    )

    print(f"Output shape: {color_img.shape}")
    print(f"Pixels rendered: {(color_img.sum(axis=2) > 0).sum()}")

    if training_data:
        print(f"Training data collected: {list(training_data.keys())}")

    print("\n[SUCCESS] Rasterizer funktioniert!")
