"""
CUDA → HIP Translator

ML-based code translator for converting CUDA to AMD HIP/ROCm.

Usage:
    from cuda_translator import translate_cuda_to_hip

    hip_code = translate_cuda_to_hip("cudaMalloc(&ptr, size);")
"""

__version__ = "0.1.0"
__all__ = ["Seq2SeqTranslator", "CharTokenizer", "translate_cuda_to_hip"]

from .translator_model import Seq2SeqTranslator, CharTokenizer


def translate_cuda_to_hip(cuda_code, model_path="cuda_translator/translator.pt"):
    """
    Translate CUDA code to HIP.

    Args:
        cuda_code: CUDA source code (string)
        model_path: Path to trained model checkpoint

    Returns:
        hip_code: Translated HIP code (string)
    """
    import torch

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # Translate
    with torch.no_grad():
        src_tokens = tokenizer.encode(cuda_code, add_sos_eos=False)
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)
        output_tokens = model.translate(src, max_len=min(len(cuda_code) * 2, 500))
        hip_code = tokenizer.decode(output_tokens[0].cpu().tolist(), skip_special=True)

    return hip_code
