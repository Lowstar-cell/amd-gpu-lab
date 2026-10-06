# AMD Compatibility Layer für ML-Projekte
# Ersetzt NVIDIA-spezifische Libraries durch AMD/ROCm Alternativen

from .flash_attention_amd import (
    flash_attn_func_amd,
    sdpa_attention,
    is_flash_attention_available,
)
from .bitsandbytes_amd import (
    ensure_bitsandbytes_rocm,
    is_bitsandbytes_available,
    get_quantization_config,
)

__version__ = "0.1.0"
__all__ = [
    "flash_attn_func_amd",
    "sdpa_attention",
    "is_flash_attention_available",
    "ensure_bitsandbytes_rocm",
    "is_bitsandbytes_available",
    "get_quantization_config",
]
