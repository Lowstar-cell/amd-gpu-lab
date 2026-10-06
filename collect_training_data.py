#!/usr/bin/env python3
"""
CUDA Training Data Collector

Sammelt CUDA Code von GitHub Repositories für Training.
"""

import subprocess
import json
from pathlib import Path
import re


class CudaDataCollector:
    """Sammelt CUDA Code für Training."""

    def __init__(self, output_dir="cuda_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.repos = []

    def add_repo(self, url, name=None):
        """Repository zur Liste hinzufügen."""
        if name is None:
            name = url.split('/')[-1].replace('.git', '')
        self.repos.append({'url': url, 'name': name})

    def clone_repos(self):
        """Alle Repositories clonen."""
        repos_dir = self.output_dir / "repos"
        repos_dir.mkdir(exist_ok=True)

        for repo in self.repos:
            repo_path = repos_dir / repo['name']

            if repo_path.exists():
                print(f"✓ {repo['name']} already cloned")
                continue

            print(f"Cloning {repo['name']}...")
            try:
                subprocess.run(
                    ['git', 'clone', '--depth', '1', repo['url'], str(repo_path)],
                    check=True,
                    capture_output=True
                )
                print(f"✓ Cloned {repo['name']}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to clone {repo['name']}: {e}")

    def extract_cuda_code(self):
        """CUDA Code aus Repositories extrahieren."""
        repos_dir = self.output_dir / "repos"
        cuda_files = []

        for repo in self.repos:
            repo_path = repos_dir / repo['name']
            if not repo_path.exists():
                continue

            # Find .cu files
            for cu_file in repo_path.rglob("*.cu"):
                # Skip test/example files that might be too simple
                if any(skip in str(cu_file) for skip in ['test', 'example', 'sample']):
                    continue

                cuda_files.append({
                    'repo': repo['name'],
                    'file': str(cu_file.relative_to(repo_path)),
                    'full_path': str(cu_file)
                })

        print(f"\nFound {len(cuda_files)} CUDA files")
        return cuda_files

    def extract_kernels(self, file_path):
        """Extract __global__ kernels from CUDA file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []

        # Find all __global__ kernel definitions
        # Pattern: __global__ void kernelName(...) { ... }
        kernel_pattern = r'__global__\s+\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\}'
        kernels = re.findall(kernel_pattern, content, re.DOTALL)

        return kernels

    def create_training_pairs(self, cuda_files, max_pairs=5000):
        """Create training pairs from CUDA files."""
        pairs = []

        for cuda_file in cuda_files[:max_pairs]:
            kernels = self.extract_kernels(cuda_file['full_path'])

            for kernel in kernels:
                # For now, just store CUDA code
                # We'll need HIPify to create actual CUDA→HIP pairs
                pairs.append({
                    'cuda': kernel.strip(),
                    'hip': None,  # Will be filled by HIPify
                    'source': f"{cuda_file['repo']}/{cuda_file['file']}"
                })

        return pairs

    def save_pairs(self, pairs, filename="cuda_code_samples.json"):
        """Save training pairs to JSON."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(pairs, f, indent=2)

        print(f"\n✓ Saved {len(pairs)} pairs to {output_file}")


def main():
    print("=" * 70)
    print("CUDA TRAINING DATA COLLECTOR")
    print("=" * 70)
    print()

    collector = CudaDataCollector()

    # Add important CUDA repositories
    repos = [
        # HIP examples (has CUDA→HIP pairs!)
        'https://github.com/ROCm-Developer-Tools/HIP-Examples.git',

        # PyTorch CUDA kernels
        'https://github.com/pytorch/pytorch.git',

        # NVIDIA Samples
        'https://github.com/NVIDIA/cuda-samples.git',

        # Thrust (CUDA algorithms)
        'https://github.com/NVIDIA/thrust.git',

        # CUB (CUDA building blocks)
        'https://github.com/NVIDIA/cub.git',
    ]

    print("Will clone the following repositories:")
    for i, repo in enumerate(repos, 1):
        name = repo.split('/')[-1].replace('.git', '')
        print(f"  {i}. {name}")
        collector.add_repo(repo)

    print()
    input("Press Enter to start cloning...")

    # Clone repositories
    print("\n" + "=" * 70)
    print("CLONING REPOSITORIES")
    print("=" * 70)
    collector.clone_repos()

    # Extract CUDA code
    print("\n" + "=" * 70)
    print("EXTRACTING CUDA CODE")
    print("=" * 70)
    cuda_files = collector.extract_cuda_code()

    # Create training pairs
    print("\n" + "=" * 70)
    print("CREATING TRAINING PAIRS")
    print("=" * 70)
    pairs = collector.create_training_pairs(cuda_files)

    # Save
    collector.save_pairs(pairs)

    print("\n" + "=" * 70)
    print("✓ COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print("1. Use HIPify to translate CUDA → HIP")
    print("2. Create final training dataset")
    print("3. Train model with new data")


if __name__ == '__main__':
    main()
