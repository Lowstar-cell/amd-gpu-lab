#!/usr/bin/env python3
"""
Quick test to verify PoC implementation.
Tests imports and basic functionality without training.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from translator_model import Seq2SeqTranslator, CharTokenizer
        print("  ✓ translator_model imports OK")
    except Exception as e:
        print(f"  ✗ translator_model import failed: {e}")
        return False

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} available")
    except Exception as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    return True


def test_tokenizer():
    """Test CharTokenizer."""
    print("\nTesting tokenizer...")
    try:
        from translator_model import CharTokenizer

        tokenizer = CharTokenizer()
        texts = ["cudaMalloc(&ptr, size);", "hipMalloc(&ptr, size);"]
        tokenizer.build_vocab(texts)

        # Encode
        tokens = tokenizer.encode("cudaMalloc", add_sos_eos=True)
        print(f"  ✓ Encoded 'cudaMalloc' to {len(tokens)} tokens")

        # Decode
        decoded = tokenizer.decode(tokens)
        print(f"  ✓ Decoded back to: '{decoded}'")

        print(f"  ✓ Vocab size: {tokenizer.vocab_size}")
        return True

    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        import torch
        from translator_model import Seq2SeqTranslator

        model = Seq2SeqTranslator(
            vocab_size=100,
            embed_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.0
        )

        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with {params:,} parameters")

        # Test forward pass
        batch_size = 2
        src_len = 10
        tgt_len = 12

        src = torch.randint(0, 100, (batch_size, src_len))
        tgt = torch.randint(0, 100, (batch_size, tgt_len))

        with torch.no_grad():
            logits = model(src, tgt, teacher_forcing_ratio=1.0)

        print(f"  ✓ Forward pass output shape: {logits.shape}")
        return True

    except Exception as e:
        print(f"  ✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data():
    """Test training data loading."""
    print("\nTesting training data...")
    try:
        import json
        from pathlib import Path

        data_path = Path("cuda_translator/training_data.json")
        if not data_path.exists():
            print(f"  ✗ Training data not found at {data_path}")
            return False

        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"  ✓ Loaded {len(data)} training examples")

        # Show first example
        if data:
            example = data[0]
            print(f"  ✓ First example: '{example['cuda']}' → '{example['hip']}'")

        return True

    except Exception as e:
        print(f"  ✗ Training data test failed: {e}")
        return False


def main():
    print("="*60)
    print("CUDA→HIP Translator PoC - Quick Test")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Tokenizer", test_tokenizer),
        ("Model Creation", test_model_creation),
        ("Training Data", test_training_data),
    ]

    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*60)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! PoC is ready.")
        print("\nNext step: Run training with:")
        print("  python3 cuda_translator/train_translator.py --epochs 100")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
