"""
AMD-kompatible Flash Attention Alternative

Ersetzt Flash Attention 2/3 durch PyTorch's native SDPA (Scaled Dot Product Attention).
SDPA ist seit PyTorch 2.0 verfügbar und nutzt automatisch optimierte Backends:
- CuDNN Attention (NVIDIA)
- Flash Attention (wenn verfügbar)
- Memory-Efficient Attention (xformers-style)
- Math Attention (Fallback)

Auf AMD ROCm wählt PyTorch automatisch das beste verfügbare Backend.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def is_flash_attention_available() -> bool:
    """
    Prüft ob native Flash Attention verfügbar ist.
    Auf AMD nutzen wir SDPA als Fallback.
    """
    try:
        from kernels import get_kernel
        _k = get_kernel("kernels-community/vllm-flash-attn3")
        return True
    except:
        pass

    try:
        import flash_attn
        return True
    except:
        pass

    return False


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch SDPA-basierte Attention.

    Args:
        q: Query tensor (B, S, H, D) oder (B, H, S, D)
        k: Key tensor (B, S, H, D) oder (B, H, S, D)
        v: Value tensor (B, S, H, D) oder (B, H, S, D)
        causal: Ob kausale Maske verwendet werden soll
        scale: Skalierungsfaktor (default: 1/sqrt(D))
        dropout_p: Dropout Wahrscheinlichkeit

    Returns:
        Attention output mit gleicher Shape wie Input
    """
    # SDPA erwartet (B, H, S, D), nicht (B, S, H, D)
    # Prüfe ob wir transponieren müssen
    input_shape = q.shape

    if len(q.shape) == 4:
        # Annahme: Input ist (B, S, H, D) - typisches FA Format
        # Wir müssen zu (B, H, S, D) transponieren
        if q.shape[1] > q.shape[2]:  # S > H typisch
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            needs_transpose_back = True
        else:
            needs_transpose_back = False
    else:
        needs_transpose_back = False

    # SDPA aufrufen
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )

    # Zurück transponieren wenn nötig
    if needs_transpose_back:
        output = output.transpose(1, 2)

    return output


def flash_attn_func_amd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Drop-in Replacement für flash_attn_func.

    Kompatibel mit dem FA2/FA3 Interface aber nutzt SDPA intern.
    Input Shape: (B, S, H, D) - Flash Attention Format
    Output Shape: (B, S, H, D)
    """
    # Input ist (B, S, H, D), SDPA braucht (B, H, S, D)
    B, S, H, D = q.shape

    q = q.transpose(1, 2).contiguous()  # (B, H, S, D)
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # SDPA aufrufen
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=softmax_scale,
    )

    # Zurück zu (B, S, H, D)
    output = output.transpose(1, 2).contiguous()

    return output


# Registriere als torch custom op für Kompatibilität mit Code der torch.library verwendet
try:
    @torch.library.custom_op("flash_amd::flash_attn_func", mutates_args=())
    def _flash_attn_func_op(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        return flash_attn_func_amd(q, k, v, causal=causal)

    @_flash_attn_func_op.register_fake
    def _(q, k, v, **kwargs):
        return torch.empty_like(q).contiguous()
except Exception:
    # Wenn torch.library nicht verfügbar ist (ältere PyTorch Version)
    pass


class SDPAAttnProcessor:
    """
    AMD-kompatibler Attention Processor basierend auf SDPA.
    Drop-in Replacement für FA2/FA3 basierte Processors.
    """

    _attention_backend = "sdpa"

    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Standard Attention mit SDPA.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        v = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)

        # Reshape zu (B, H, S, D)
        head_dim = q.shape[-1] // attn.heads
        q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # SDPA
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,
        )

        # Reshape zurück
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # Output projection
        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output
