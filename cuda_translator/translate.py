#!/usr/bin/env python3
"""
Translate CUDA code to HIP using trained model.

Usage:
    # Translate single snippet
    python translate.py "cudaMalloc(&ptr, size);"

    # Translate file
    python translate.py --input kernel.cu --output kernel.hip

    # Interactive mode
    python translate.py --interactive
"""

import argparse
import sys
from pathlib import Path
import torch
from translator_model import Seq2SeqTranslator, CharTokenizer


def load_model(model_path, device):
    """Load trained translator model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Restore tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char2id = checkpoint['tokenizer_char2id']
    tokenizer.id2char = checkpoint['tokenizer_id2char']
    tokenizer.next_id = checkpoint['vocab_size']

    # Restore model (dropout=0.0 wegen ROCm/MIOpen Bug)
    model = Seq2SeqTranslator(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer


def translate_code(model, tokenizer, cuda_code, device):
    """Translate CUDA code to HIP."""
    with torch.no_grad():
        # Encode
        src_tokens = tokenizer.encode(cuda_code, add_sos_eos=False)
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)

        # Translate
        output_tokens = model.translate(src, max_len=min(len(cuda_code) * 2, 500))

        # Decode
        hip_code = tokenizer.decode(output_tokens[0].cpu().tolist(), skip_special=True)

    return hip_code


def translate_file(model, tokenizer, input_path, output_path, device):
    """Translate entire CUDA file to HIP."""
    with open(input_path, 'r') as f:
        cuda_code = f.read()

    # Translate line by line for better results
    lines = cuda_code.split('\n')
    translated_lines = []

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            translated_lines.append(line)
            continue

        # Translate line
        hip_line = translate_code(model, tokenizer, line, device)
        translated_lines.append(hip_line)

    hip_code = '\n'.join(translated_lines)

    # Write output
    if output_path:
        with open(output_path, 'w') as f:
            f.write(hip_code)
        print(f"Translation saved to: {output_path}")
    else:
        print(hip_code)

    return hip_code


def interactive_mode(model, tokenizer, device):
    """Interactive translation mode."""
    print("CUDA → HIP Translator (Interactive Mode)")
    print("Enter CUDA code (or 'quit' to exit):")
    print("-" * 60)

    while True:
        try:
            cuda_code = input("\nCUDA> ").strip()

            if cuda_code.lower() in ['quit', 'exit', 'q']:
                break

            if not cuda_code:
                continue

            hip_code = translate_code(model, tokenizer, cuda_code, device)
            print(f"HIP>  {hip_code}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='CUDA → HIP Translator')
    parser.add_argument('code', nargs='?', help='CUDA code snippet to translate')
    parser.add_argument('--model', default='cuda_translator/translator.pt', help='Model checkpoint path')
    parser.add_argument('--input', '-i', help='Input CUDA file')
    parser.add_argument('--output', '-o', help='Output HIP file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', default='auto', help='Device (cpu/cuda/auto)')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, device)
    print(f"Model loaded (vocab_size={tokenizer.vocab_size}, device={device})")

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, device)
        return

    # File translation
    if args.input:
        translate_file(model, tokenizer, args.input, args.output, device)
        return

    # Single snippet
    if args.code:
        hip_code = translate_code(model, tokenizer, args.code, device)
        print(f"CUDA: {args.code}")
        print(f"HIP:  {hip_code}")
        return

    # No input provided
    parser.print_help()


if __name__ == '__main__':
    main()
