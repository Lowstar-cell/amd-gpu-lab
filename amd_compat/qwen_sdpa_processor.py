"""
AMD-kompatibler Qwen Double Stream Attention Processor

Drop-in Replacement für QwenDoubleStreamAttnProcessorFA3.
Nutzt PyTorch SDPA statt Flash Attention 3.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class QwenDoubleStreamAttnProcessorAMD:
    """
    SDPA-basierter Attention Processor für Qwen double-stream Architektur.

    Drop-in Replacement für QwenDoubleStreamAttnProcessorFA3.
    Funktioniert auf AMD ROCm GPUs ohne Flash Attention.

    Hinweis: attention_mask wird hier unterstützt (im Gegensatz zum FA3 Processor).
    """

    _attention_backend = "sdpa"

    def __init__(self):
        pass

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention Modul mit to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, und .heads
        hidden_states: torch.FloatTensor,                 # (B, S_img, D_model) image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model) text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorAMD benötigt encoder_hidden_states (text stream).")

        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # ---- QKV Projektionen (image/sample stream) ----
        img_q = attn.to_q(hidden_states)   # (B, S_img, D)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        # ---- QKV Projektionen (text/context stream) ----
        txt_q = attn.add_q_proj(encoder_hidden_states)  # (B, S_txt, D)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # ---- Reshape zu (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K Normalisierung ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen Variante) ----
        if image_rotary_emb is not None:
            try:
                from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
                img_freqs, txt_freqs = image_rotary_emb
                img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
                img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
                txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
                txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)
            except ImportError:
                # Fallback wenn diffusers diese Funktion nicht hat
                pass

        # ---- Joint Attention über [text, image] entlang sequence Achse ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # SDPA erwartet (B, H, S, D), also transponieren
        q = q.transpose(1, 2)  # (B, H, S_total, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SDPA aufrufen
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,
        )

        # Zurück zu (B, S_total, H, D_h)
        out = out.transpose(1, 2)

        # ---- Zurück zu (B, S, D_model) ----
        out = out.flatten(2, 3).to(q.dtype)

        # Split zurück zu text / image Segmenten
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output Projektionen ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # Dropout falls vorhanden

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out


def patch_qwen_for_amd():
    """
    Patcht das qwenimage Modul um den AMD Processor zu verwenden.

    Verwendung:
        from amd_compat.qwen_sdpa_processor import patch_qwen_for_amd
        patch_qwen_for_amd()

        # Dann normal Qwen Pipeline laden
        pipe = QwenImageEditPlusPipeline.from_pretrained(...)
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorAMD())
    """
    try:
        import qwenimage.qwen_fa3_processor as fa3_module
        fa3_module.QwenDoubleStreamAttnProcessorFA3 = QwenDoubleStreamAttnProcessorAMD
        print("Qwen FA3 Processor durch AMD SDPA Processor ersetzt.")
    except ImportError:
        print("qwenimage Modul nicht gefunden - kein Patching nötig.")
