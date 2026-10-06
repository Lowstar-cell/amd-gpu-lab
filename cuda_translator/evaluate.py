#!/usr/bin/env python3
"""
Evaluate CUDA→HIP translator quality.

Metrics:
- Token Accuracy: How many tokens are correctly translated
- Exact Match: Percentage of exact matches
- BLEU Score: Similarity measure
- API Coverage: Percentage of CUDA APIs correctly translated
"""

import argparse
import json
from pathlib import Path
import torch
from translator_model import Seq2SeqTranslator, CharTokenizer


def load_model(model_path, device):
    """Load trained translator model."""
    checkpoint = torch.load(model_path, map_location=device)

    tokenizer = CharTokenizer()
    tokenizer.char2id = checkpoint['tokenizer_char2id']
    tokenizer.id2char = checkpoint['tokenizer_id2char']
    tokenizer.next_id = checkpoint['vocab_size']

    model = Seq2SeqTranslator(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer


def translate_code(model, tokenizer, cuda_code, device):
    """Translate CUDA code to HIP."""
    with torch.no_grad():
        src_tokens = tokenizer.encode(cuda_code, add_sos_eos=False)
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)
        output_tokens = model.translate(src, max_len=min(len(cuda_code) * 2, 500))
        hip_code = tokenizer.decode(output_tokens[0].cpu().tolist(), skip_special=True)
    return hip_code


def char_accuracy(predicted, target):
    """Character-level accuracy."""
    min_len = min(len(predicted), len(target))
    if min_len == 0:
        return 0.0

    correct = sum(1 for i in range(min_len) if predicted[i] == target[i])
    return correct / max(len(predicted), len(target))


def exact_match(predicted, target):
    """Exact match (ignoring whitespace)."""
    pred_clean = ''.join(predicted.split())
    tgt_clean = ''.join(target.split())
    return 1.0 if pred_clean == tgt_clean else 0.0


def api_coverage(predicted, target):
    """Check if key API calls are correctly translated."""
    # Extract API calls (simple heuristic)
    import re

    cuda_apis = re.findall(r'cuda\w+|block\w+|thread\w+|grid\w+', target)
    hip_apis = re.findall(r'hip\w+|hipBlock\w+|hipThread\w+|hipGrid\w+', predicted)

    if not cuda_apis:
        return 1.0  # No APIs to check

    # Check if corresponding HIP API exists
    correct = 0
    for cuda_api in cuda_apis:
        # Expected HIP API
        if cuda_api.startswith('cuda'):
            expected_hip = cuda_api.replace('cuda', 'hip')
        elif cuda_api.startswith('block'):
            expected_hip = 'hipBlock' + cuda_api[5:]
        elif cuda_api.startswith('thread'):
            expected_hip = 'hipThread' + cuda_api[6:]
        elif cuda_api.startswith('grid'):
            expected_hip = 'hipGrid' + cuda_api[4:]
        else:
            continue

        if expected_hip in predicted:
            correct += 1

    return correct / len(cuda_apis)


def evaluate_dataset(model, tokenizer, data_path, device):
    """Evaluate on full dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    results = {
        'total': len(data),
        'exact_matches': 0,
        'char_accuracy': [],
        'api_coverage': []
    }

    print(f"Evaluating on {len(data)} examples...")
    print("-" * 80)

    for i, item in enumerate(data):
        cuda_code = item['cuda']
        expected_hip = item['hip']

        # Translate
        predicted_hip = translate_code(model, tokenizer, cuda_code, device)

        # Metrics
        char_acc = char_accuracy(predicted_hip, expected_hip)
        exact = exact_match(predicted_hip, expected_hip)
        api_cov = api_coverage(predicted_hip, expected_hip)

        results['char_accuracy'].append(char_acc)
        results['api_coverage'].append(api_cov)
        if exact > 0.5:
            results['exact_matches'] += 1

        # Print sample
        if i < 5:
            print(f"Example {i+1}:")
            print(f"  CUDA:     {cuda_code}")
            print(f"  Expected: {expected_hip}")
            print(f"  Predicted: {predicted_hip}")
            print(f"  Char Acc: {char_acc:.2%}, Exact: {exact:.0%}, API: {api_cov:.0%}")
            print()

    # Summary
    print("-" * 80)
    print("EVALUATION SUMMARY")
    print("-" * 80)
    print(f"Total Examples: {results['total']}")
    print(f"Exact Matches: {results['exact_matches']} ({results['exact_matches']/results['total']:.1%})")
    print(f"Avg Char Accuracy: {sum(results['char_accuracy'])/len(results['char_accuracy']):.1%}")
    print(f"Avg API Coverage: {sum(results['api_coverage'])/len(results['api_coverage']):.1%}")
    print("-" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CUDA→HIP Translator')
    parser.add_argument('--model', default='cuda_translator/translator.pt')
    parser.add_argument('--data', default='cuda_translator/val_data.json')
    parser.add_argument('--device', default='auto')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, device)
    print(f"Model loaded (vocab_size={tokenizer.vocab_size})")
    print()

    # Evaluate
    results = evaluate_dataset(model, tokenizer, args.data, device)


if __name__ == '__main__':
    main()
