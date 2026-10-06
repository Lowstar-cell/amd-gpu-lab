#!/usr/bin/env python3
"""
CUDA→HIP Training Data Collection Pipeline

Automatisch CUDA-Code von GitHub holen, mit HIPify konvertieren,
und Trainingspaare extrahieren.

Usage:
    python collect_training_data.py --output training_data_collected.json
    python collect_training_data.py --repos repos.txt --output data.json
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import shutil

# Standard CUDA-Repos zum Klonen
DEFAULT_REPOS = [
    "https://github.com/NVIDIA/cuda-samples.git",
    "https://github.com/bryancatanzaro/inplace.git",
    "https://github.com/parallel-forall/code-samples.git",
]

# Einfache CUDA→HIP API Mappings (Fallback wenn hipify nicht klappt)
CUDA_TO_HIP_SIMPLE = {
    'cudaMalloc': 'hipMalloc',
    'cudaFree': 'hipFree',
    'cudaMemcpy': 'hipMemcpy',
    'cudaMemcpyAsync': 'hipMemcpyAsync',
    'cudaMemset': 'hipMemset',
    'cudaMemsetAsync': 'hipMemsetAsync',
    'cudaDeviceSynchronize': 'hipDeviceSynchronize',
    'cudaStreamSynchronize': 'hipStreamSynchronize',
    'cudaStreamCreate': 'hipStreamCreate',
    'cudaStreamDestroy': 'hipStreamDestroy',
    'cudaEventCreate': 'hipEventCreate',
    'cudaEventDestroy': 'hipEventDestroy',
    'cudaEventRecord': 'hipEventRecord',
    'cudaEventSynchronize': 'hipEventSynchronize',
    'cudaEventElapsedTime': 'hipEventElapsedTime',
    'cudaGetDeviceCount': 'hipGetDeviceCount',
    'cudaGetDevice': 'hipGetDevice',
    'cudaSetDevice': 'hipSetDevice',
    'cudaGetDeviceProperties': 'hipGetDeviceProperties',
    'cudaMemcpyHostToDevice': 'hipMemcpyHostToDevice',
    'cudaMemcpyDeviceToHost': 'hipMemcpyDeviceToHost',
    'cudaMemcpyDeviceToDevice': 'hipMemcpyDeviceToDevice',
    'cudaMemcpyHostToHost': 'hipMemcpyHostToHost',
    'cudaSuccess': 'hipSuccess',
    'cudaError_t': 'hipError_t',
    'cudaStream_t': 'hipStream_t',
    'cudaEvent_t': 'hipEvent_t',
    'cudaDeviceProp': 'hipDeviceProp_t',
    '__syncthreads': '__syncthreads',  # bleibt gleich
    'blockIdx': 'blockIdx',  # bleibt gleich in HIP
    'blockDim': 'blockDim',
    'threadIdx': 'threadIdx',
    'gridDim': 'gridDim',
}


def clone_repo(url: str, target_dir: Path) -> bool:
    """Clone a git repository."""
    print(f"  Klone {url}...")
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        print(f"    Fehler: {e}")
        return False


def find_cuda_files(directory: Path) -> List[Path]:
    """Find all CUDA source files."""
    cuda_files = []
    for ext in ['*.cu', '*.cuh']:
        cuda_files.extend(directory.rglob(ext))
    return cuda_files


def hipify_file(cuda_file: Path) -> str:
    """Convert CUDA file to HIP using hipify-clang."""
    try:
        result = subprocess.run(
            ['hipify-clang', str(cuda_file), '--print-stats=false', '-o', '-'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""


def simple_hipify(cuda_code: str) -> str:
    """Simple regex-based CUDA→HIP conversion as fallback."""
    hip_code = cuda_code
    for cuda, hip in CUDA_TO_HIP_SIMPLE.items():
        hip_code = re.sub(r'\b' + cuda + r'\b', hip, hip_code)
    return hip_code


def extract_pairs_from_lines(cuda_lines: List[str], hip_lines: List[str], max_len: int = 300) -> List[Dict]:
    """Extract CUDA→HIP pairs from matching lines."""
    pairs = []

    for cuda_line, hip_line in zip(cuda_lines, hip_lines):
        cuda_line = cuda_line.strip()
        hip_line = hip_line.strip()

        # Skip empty lines or comments
        if not cuda_line or not hip_line:
            continue
        if cuda_line.startswith('//') or cuda_line.startswith('/*'):
            continue
        if cuda_line.startswith('#include'):
            continue

        # Skip if identical (no translation needed)
        if cuda_line == hip_line:
            continue

        # Skip if too long
        if len(cuda_line) > max_len or len(hip_line) > max_len:
            continue

        # Must contain cuda/hip keywords to be interesting
        if 'cuda' not in cuda_line.lower() and '__global__' not in cuda_line:
            continue

        pairs.append({
            'cuda': cuda_line,
            'hip': hip_line
        })

    return pairs


def extract_api_calls(cuda_code: str, max_len: int = 300) -> List[Dict]:
    """Extract individual API calls and convert them."""
    pairs = []

    # Pattern für CUDA API calls
    patterns = [
        r'cuda\w+\s*\([^;]+\);',  # cudaXxx(...);
        r'cuda\w+\s*<<<[^>]+>>>\s*\([^;]+\);',  # kernel<<<...>>>(...);
    ]

    for pattern in patterns:
        matches = re.findall(pattern, cuda_code)
        for match in matches:
            cuda_call = match.strip()
            if len(cuda_call) > max_len:
                continue

            hip_call = simple_hipify(cuda_call)

            if cuda_call != hip_call:
                pairs.append({
                    'cuda': cuda_call,
                    'hip': hip_call
                })

    return pairs


def process_cuda_file(cuda_file: Path, use_hipify: bool = True) -> List[Dict]:
    """Process a single CUDA file and extract training pairs."""
    pairs = []

    try:
        with open(cuda_file, 'r', encoding='utf-8', errors='ignore') as f:
            cuda_code = f.read()
    except Exception:
        return pairs

    # Method 1: Use hipify-clang
    if use_hipify:
        hip_code = hipify_file(cuda_file)
        if hip_code:
            cuda_lines = cuda_code.split('\n')
            hip_lines = hip_code.split('\n')
            pairs.extend(extract_pairs_from_lines(cuda_lines, hip_lines))

    # Method 2: Extract API calls with simple conversion
    pairs.extend(extract_api_calls(cuda_code))

    return pairs


def deduplicate_pairs(pairs: List[Dict]) -> List[Dict]:
    """Remove duplicate pairs."""
    seen = set()
    unique = []

    for pair in pairs:
        key = (pair['cuda'], pair['hip'])
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    return unique


def collect_from_repos(repos: List[str], work_dir: Path, use_hipify: bool = True) -> List[Dict]:
    """Collect training data from multiple repositories."""
    all_pairs = []

    for repo_url in repos:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_dir = work_dir / repo_name

        print(f"\n[{repo_name}]")

        # Clone
        if not repo_dir.exists():
            if not clone_repo(repo_url, repo_dir):
                print(f"  Überspringe (clone fehlgeschlagen)")
                continue

        # Find CUDA files
        cuda_files = find_cuda_files(repo_dir)
        print(f"  Gefunden: {len(cuda_files)} CUDA-Dateien")

        # Process files
        repo_pairs = []
        for cuda_file in cuda_files:
            pairs = process_cuda_file(cuda_file, use_hipify)
            repo_pairs.extend(pairs)

        print(f"  Extrahiert: {len(repo_pairs)} Paare")
        all_pairs.extend(repo_pairs)

    return all_pairs


def collect_from_local(directory: Path, use_hipify: bool = True) -> List[Dict]:
    """Collect training data from local CUDA files."""
    all_pairs = []

    cuda_files = find_cuda_files(directory)
    print(f"Gefunden: {len(cuda_files)} CUDA-Dateien in {directory}")

    for cuda_file in cuda_files:
        pairs = process_cuda_file(cuda_file, use_hipify)
        all_pairs.extend(pairs)

    return all_pairs


def main():
    parser = argparse.ArgumentParser(description='CUDA→HIP Training Data Pipeline')
    parser.add_argument('--output', '-o', default='training_data_collected.json',
                        help='Output JSON file')
    parser.add_argument('--repos', '-r', help='File with repo URLs (one per line)')
    parser.add_argument('--local', '-l', help='Local directory with CUDA files')
    parser.add_argument('--work-dir', '-w', default='/tmp/cuda_repos',
                        help='Working directory for cloned repos')
    parser.add_argument('--no-hipify', action='store_true',
                        help='Skip hipify-clang (use simple conversion only)')
    parser.add_argument('--max-len', type=int, default=300,
                        help='Maximum line length for pairs')
    parser.add_argument('--min-pairs', type=int, default=10,
                        help='Minimum pairs to consider successful')

    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    use_hipify = not args.no_hipify

    print("=" * 60)
    print("CUDA→HIP Training Data Pipeline")
    print("=" * 60)

    all_pairs = []

    # Collect from repos
    if args.repos:
        with open(args.repos) as f:
            repos = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        repos = DEFAULT_REPOS

    if not args.local:
        print(f"\nRepos: {len(repos)}")
        all_pairs.extend(collect_from_repos(repos, work_dir, use_hipify))

    # Collect from local directory
    if args.local:
        local_dir = Path(args.local)
        if local_dir.exists():
            all_pairs.extend(collect_from_local(local_dir, use_hipify))

    # Also check existing cuda_training_data
    existing_data = Path(__file__).parent.parent / 'cuda_training_data'
    if existing_data.exists():
        print(f"\nLokal: {existing_data}")
        all_pairs.extend(collect_from_local(existing_data, use_hipify))

    # Deduplicate
    print(f"\n" + "=" * 60)
    print(f"Gesamt vor Deduplizierung: {len(all_pairs)}")
    unique_pairs = deduplicate_pairs(all_pairs)
    print(f"Gesamt nach Deduplizierung: {len(unique_pairs)}")

    # Filter by length
    short_pairs = [p for p in unique_pairs if len(p['cuda']) <= args.max_len and len(p['hip']) <= args.max_len]
    print(f"Kurze Paare (≤{args.max_len} chars): {len(short_pairs)}")

    # Save
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    with open(output_path, 'w') as f:
        json.dump(short_pairs, f, indent=2)

    print(f"\nGespeichert: {output_path}")
    print(f"Paare: {len(short_pairs)}")

    # Show examples
    if short_pairs:
        print(f"\nBeispiele:")
        for pair in short_pairs[:5]:
            print(f"  CUDA: {pair['cuda'][:60]}...")
            print(f"  HIP:  {pair['hip'][:60]}...")
            print()

    if len(short_pairs) < args.min_pairs:
        print(f"\n⚠ Weniger als {args.min_pairs} Paare gesammelt!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
