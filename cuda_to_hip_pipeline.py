#!/usr/bin/env python3
"""
CUDA → HIP Translation Pipeline

Complete pipeline for translating CUDA code to HIP/ROCm.
"""

import argparse
import sys
from pathlib import Path
import re

# Add cuda_translator to path
sys.path.insert(0, str(Path(__file__).parent / "cuda_translator"))

from translator_model import Seq2SeqTranslator, CharTokenizer
import torch


class CudaToHipPipeline:
    """Complete CUDA → HIP translation pipeline."""

    def __init__(self, model_path="cuda_translator/translator.pt", device="cpu"):
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Reconstruct tokenizer
        self.tokenizer = CharTokenizer()
        self.tokenizer.char2id = checkpoint['tokenizer_char2id']
        self.tokenizer.id2char = checkpoint['tokenizer_id2char']
        self.tokenizer.vocab = list(self.tokenizer.char2id.keys())

        self.model = Seq2SeqTranslator(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint.get('embed_dim', 128),
            hidden_dim=checkpoint.get('hidden_dim', 256),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=0.0  # No dropout during inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✓ Model loaded")

    def translate_snippet(self, cuda_code, max_length=200):
        """Translate a single CUDA code snippet to HIP."""
        with torch.no_grad():
            # Encode input
            input_ids = self.tokenizer.encode(cuda_code)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            # Translate
            output_ids = self.model.translate(input_tensor, max_len=max_length)[0]

            # Decode output
            hip_code = self.tokenizer.decode(output_ids)

            return hip_code

    def translate_file(self, cuda_file, hip_file=None, preserve_comments=True):
        """Translate entire CUDA file to HIP."""

        # Read CUDA code
        with open(cuda_file, 'r') as f:
            cuda_code = f.read()

        # Split into logical chunks (functions, kernels, etc.)
        chunks = self.split_code(cuda_code)

        # Translate each chunk
        hip_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk['type'] == 'code':
                print(f"Translating chunk {i+1}/{len(chunks)}...", end='\r')
                translated = self.translate_snippet(chunk['content'])
                hip_chunks.append(translated)
            else:
                # Preserve comments, includes, etc.
                hip_chunks.append(chunk['content'])

        print()
        hip_code = '\n'.join(hip_chunks)

        # Post-processing: apply common CUDA → HIP rules
        hip_code = self.apply_rules(hip_code)

        # Save or return
        if hip_file:
            with open(hip_file, 'w') as f:
                f.write(hip_code)
            print(f"✓ Saved to {hip_file}")

        return hip_code

    def split_code(self, code):
        """Split code into chunks for translation."""
        # Simple chunking: split by lines for now
        # Could be improved to split by functions, kernels, etc.
        lines = code.split('\n')
        chunks = []

        for line in lines:
            # Detect if line is a comment or preprocessor directive
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#') or not stripped:
                chunks.append({'type': 'preserve', 'content': line})
            else:
                chunks.append({'type': 'code', 'content': line})

        return chunks

    def apply_rules(self, code):
        """Apply deterministic CUDA → HIP translation rules."""

        rules = [
            # Include headers
            (r'#include\s+<cuda\.h>', '#include <hip/hip_runtime.h>'),
            (r'#include\s+<cuda_runtime\.h>', '#include <hip/hip_runtime.h>'),

            # API calls
            (r'\bcudaMalloc\b', 'hipMalloc'),
            (r'\bcudaFree\b', 'hipFree'),
            (r'\bcudaMemcpy\b', 'hipMemcpy'),
            (r'\bcudaMemcpyHostToDevice\b', 'hipMemcpyHostToDevice'),
            (r'\bcudaMemcpyDeviceToHost\b', 'hipMemcpyDeviceToHost'),
            (r'\bcudaMemset\b', 'hipMemset'),
            (r'\bcudaDeviceSynchronize\b', 'hipDeviceSynchronize'),
            (r'\bcudaGetLastError\b', 'hipGetLastError'),
            (r'\bcudaGetErrorString\b', 'hipGetErrorString'),
            (r'\bcudaSuccess\b', 'hipSuccess'),

            # Thread indexing
            (r'\bthreadIdx\.x\b', 'hipThreadIdx_x'),
            (r'\bthreadIdx\.y\b', 'hipThreadIdx_y'),
            (r'\bthreadIdx\.z\b', 'hipThreadIdx_z'),
            (r'\bblockIdx\.x\b', 'hipBlockIdx_x'),
            (r'\bblockIdx\.y\b', 'hipBlockIdx_y'),
            (r'\bblockIdx\.z\b', 'hipBlockIdx_z'),
            (r'\bblockDim\.x\b', 'hipBlockDim_x'),
            (r'\bblockDim\.y\b', 'hipBlockDim_y'),
            (r'\bblockDim\.z\b', 'hipBlockDim_z'),
            (r'\bgridDim\.x\b', 'hipGridDim_x'),
            (r'\bgridDim\.y\b', 'hipGridDim_y'),
            (r'\bgridDim\.z\b', 'hipGridDim_z'),

            # Kernel attributes
            (r'\b__global__\b', '__global__'),  # Same in HIP
            (r'\b__device__\b', '__device__'),  # Same in HIP
            (r'\b__host__\b', '__host__'),      # Same in HIP

            # Kernel launch syntax (keep same)
            # <<<>>> syntax is the same
        ]

        for pattern, replacement in rules:
            code = re.sub(pattern, replacement, code)

        return code

    def validate_hip(self, hip_code):
        """Basic validation of HIP code."""
        errors = []

        # Check for remaining CUDA-specific terms
        cuda_terms = ['cudaMalloc', 'cudaFree', 'threadIdx.x', 'blockIdx.x']
        for term in cuda_terms:
            if term in hip_code:
                errors.append(f"Found untranslated CUDA term: {term}")

        # Check for HIP includes
        if '#include' in hip_code and 'hip_runtime' not in hip_code:
            errors.append("Missing HIP runtime include")

        return errors


def main():
    parser = argparse.ArgumentParser(description='CUDA → HIP Translation Pipeline')
    parser.add_argument('input', nargs='?', help='Input CUDA file (or stdin if not provided)')
    parser.add_argument('-o', '--output', help='Output HIP file (or stdout if not provided)')
    parser.add_argument('--model', default='cuda_translator/translator.pt', help='Path to trained model')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    parser.add_argument('--validate', action='store_true', help='Validate output HIP code')
    parser.add_argument('--snippet', action='store_true', help='Translate single snippet instead of file')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = CudaToHipPipeline(model_path=args.model, device=args.device)

    print("=" * 70)
    print("CUDA → HIP TRANSLATION PIPELINE")
    print("=" * 70)
    print()

    # Read input
    if args.input:
        if args.snippet:
            with open(args.input, 'r') as f:
                cuda_code = f.read()
            hip_code = pipeline.translate_snippet(cuda_code)
        else:
            hip_code = pipeline.translate_file(args.input, args.output)
    else:
        # Read from stdin
        print("Enter CUDA code (Ctrl+D to finish):")
        cuda_code = sys.stdin.read()
        hip_code = pipeline.translate_snippet(cuda_code)

    # Validate if requested
    if args.validate:
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)
        errors = pipeline.validate_hip(hip_code)
        if errors:
            print("⚠ Warnings:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✓ No obvious issues found")

    # Output
    if not args.output:
        print("\n" + "=" * 70)
        print("HIP OUTPUT")
        print("=" * 70)
        print(hip_code)

    print()
    print("=" * 70)
    print("✓ Translation complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
