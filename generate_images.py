#!/usr/bin/env python3
"""
Schnelle Bildgenerierung für AMD GPU Demo

Erzeugt:
- Fraktale (Mandelbrot)
- Noise Patterns
- Gradient Art
- Procedural Textures
"""

import numpy as np
import torch
from pathlib import Path


def save_image_png(array, filename):
    """Save numpy array as PNG (ohne matplotlib dependency)."""
    try:
        from PIL import Image
        # Normalize to 0-255
        if array.dtype != np.uint8:
            array = np.clip(array * 255, 0, 255).astype(np.uint8)

        # Convert to PIL and save
        if len(array.shape) == 2:
            img = Image.fromarray(array, mode='L')
        else:
            img = Image.fromarray(array, mode='RGB')

        img.save(filename)
        print(f"  ✓ Saved: {filename}")
        return True
    except ImportError:
        # Fallback: save as numpy
        np.save(filename.replace('.png', '.npy'), array)
        print(f"  ✓ Saved (numpy): {filename.replace('.png', '.npy')}")
        return False


def generate_mandelbrot(width=1024, height=1024, max_iter=100, device='cpu'):
    """Generate Mandelbrot fractal on GPU/CPU."""
    print(f"Generating Mandelbrot ({width}x{height}) on {device}...")

    # Create coordinate grid
    x = torch.linspace(-2.5, 1.5, width, device=device)
    y = torch.linspace(-1.5, 1.5, height, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Complex plane
    C = X + 1j * Y
    Z = torch.zeros_like(C)
    M = torch.zeros(C.shape, device=device)

    # Iterate
    for i in range(max_iter):
        mask = (Z.abs() <= 2)
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] = i

    # Normalize and convert to numpy
    img = M.cpu().numpy().astype(np.float32) / max_iter

    # Create colorful version (RGB)
    img_rgb = np.stack([
        np.sin(img * 3.14159 * 2) ** 2,
        np.sin(img * 3.14159 * 3) ** 2,
        np.sin(img * 3.14159 * 5) ** 2,
    ], axis=-1)

    return img_rgb


def generate_perlin_noise(width=1024, height=1024, scale=10, device='cpu'):
    """Generate Perlin-like noise pattern."""
    print(f"Generating noise pattern ({width}x{height}) on {device}...")

    # Multi-octave noise
    noise = torch.zeros(height, width, device=device)

    for octave in range(5):
        freq = 2 ** octave
        amp = 1 / (2 ** octave)

        # Generate random noise at this frequency
        size = (height // freq + 1, width // freq + 1)
        octave_noise = torch.rand(size, device=device) * amp

        # Upsample
        octave_noise = torch.nn.functional.interpolate(
            octave_noise.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=True
        ).squeeze()

        noise += octave_noise

    # Normalize
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Make colorful
    img_rgb = np.stack([
        noise.cpu().numpy(),
        (1 - noise).cpu().numpy(),
        (noise * 2 % 1).cpu().numpy(),
    ], axis=-1)

    return img_rgb


def generate_gradient_art(width=1024, height=1024, device='cpu'):
    """Generate gradient art with circles."""
    print(f"Generating gradient art ({width}x{height}) on {device}...")

    # Create coordinate grids
    x = torch.linspace(-1, 1, width, device=device)
    y = torch.linspace(-1, 1, height, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Multiple circles with gradients
    img = torch.zeros(height, width, 3, device=device)

    for i in range(5):
        # Random circle center
        cx = (torch.rand(1, device=device) - 0.5) * 2
        cy = (torch.rand(1, device=device) - 0.5) * 2

        # Distance from center
        dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)

        # Color based on distance
        color = torch.sin(dist * 10 + i) ** 2

        # Add to each channel with phase shift
        img[:, :, 0] += color * torch.sin(i)
        img[:, :, 1] += color * torch.cos(i)
        img[:, :, 2] += color * torch.sin(i + 1)

    # Normalize
    img = (img - img.min()) / (img.max() - img.min())

    return img.cpu().numpy()


def generate_julia_set(width=1024, height=1024, c_real=-0.7, c_imag=0.27015, max_iter=100, device='cpu'):
    """Generate Julia set fractal."""
    print(f"Generating Julia set ({width}x{height}) on {device}...")

    # Create coordinate grid
    x = torch.linspace(-1.5, 1.5, width, device=device)
    y = torch.linspace(-1.5, 1.5, height, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Complex plane
    Z = X + 1j * Y
    c = complex(c_real, c_imag)
    M = torch.zeros(Z.shape, device=device)

    # Iterate
    for i in range(max_iter):
        mask = (Z.abs() <= 2)
        Z[mask] = Z[mask] ** 2 + c
        M[mask] = i

    # Normalize
    img = M.cpu().numpy().astype(np.float32) / max_iter

    # Colorize
    img_rgb = np.stack([
        np.sin(img * 3.14159 * 4) ** 2,
        img,
        np.cos(img * 3.14159 * 4) ** 2,
    ], axis=-1)

    return img_rgb


def generate_plasma(width=1024, height=1024, device='cpu'):
    """Generate plasma effect."""
    print(f"Generating plasma ({width}x{height}) on {device}...")

    x = torch.linspace(0, 4 * 3.14159, width, device=device)
    y = torch.linspace(0, 4 * 3.14159, height, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Multiple sine waves
    plasma = torch.sin(X) + torch.sin(Y)
    plasma += torch.sin((X + Y) / 2)
    plasma += torch.sin(torch.sqrt(X**2 + Y**2) / 2)

    # Normalize
    plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min())

    # Rainbow colors
    img_rgb = np.stack([
        torch.sin(plasma * 3.14159 * 2).cpu().numpy(),
        torch.sin(plasma * 3.14159 * 2 + 2).cpu().numpy(),
        torch.sin(plasma * 3.14159 * 2 + 4).cpu().numpy(),
    ], axis=-1)

    return img_rgb


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate beautiful images')
    parser.add_argument('--size', type=int, default=1024, help='Image size')
    parser.add_argument('--device', default='auto', help='cpu/cuda/auto')
    parser.add_argument('--output', default='generated_images', help='Output directory')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print(f"GPU Image Generator")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # Generate images
    generators = [
        ('mandelbrot', lambda: generate_mandelbrot(args.size, args.size, device=device)),
        ('julia', lambda: generate_julia_set(args.size, args.size, device=device)),
        ('plasma', lambda: generate_plasma(args.size, args.size, device=device)),
        ('noise', lambda: generate_perlin_noise(args.size, args.size, device=device)),
        ('gradient', lambda: generate_gradient_art(args.size, args.size, device=device)),
    ]

    import time
    total_time = 0

    for name, gen_func in generators:
        print(f"\n[{name.upper()}]")
        start = time.time()

        try:
            img = gen_func()
            elapsed = time.time() - start
            total_time += elapsed

            print(f"  Generation time: {elapsed:.3f}s")

            # Save
            filename = out_dir / f"{name}_{args.size}x{args.size}.png"
            save_image_png(img, str(filename))

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"✓ Generated {len(generators)} images in {total_time:.2f}s")
    print(f"✓ Saved to: {out_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
