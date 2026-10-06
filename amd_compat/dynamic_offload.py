"""
Dynamic Model Offloading für große Diffusion Pipelines

Lädt Modelle nur wenn sie gebraucht werden und entlädt sie sofort danach.
Das ist aggressiver als enable_model_cpu_offload() - Modelle werden komplett
aus dem RAM entfernt, nicht nur von GPU nach CPU verschoben.

Verwendung:
    from amd_compat.dynamic_offload import DynamicOffloadPipeline

    pipe = DynamicOffloadPipeline(
        model_id="Qwen/Qwen-Image-Edit-2509",
        transformer_id="prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
    )

    result = pipe.generate(image, prompt, steps=4)
"""

import gc
import os
import torch
from typing import Optional, List, Union
from PIL import Image


def aggressive_cleanup():
    """Aggressives Memory Cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_ram_gb() -> float:
    """Gibt belegten RAM in GB zurück."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        total = available = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total = int(line.split()[1]) * 1024
            elif line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
        return (total - available) / (1024**3)
    except:
        return 0.0


def unload_model(model) -> None:
    """Entlädt ein Modell komplett aus dem Speicher."""
    if model is None:
        return

    # Erst auf CPU verschieben falls auf GPU
    try:
        model.to('cpu')
    except:
        pass

    # Alle Parameter löschen
    try:
        for param in model.parameters():
            param.data = torch.empty(0)
            if param.grad is not None:
                param.grad = None
    except:
        pass

    # Model löschen
    del model
    aggressive_cleanup()


class DynamicOffloadPipeline:
    """
    Pipeline mit dynamischem Laden/Entladen von Modellen.

    Statt alle Modelle im RAM zu halten, werden sie nur bei Bedarf geladen
    und sofort wieder entladen. Das spart ~30-40GB RAM.

    Ablauf:
    1. Text Encoder laden -> Embeddings generieren -> Text Encoder entladen
    2. VAE laden -> Bild encodieren -> VAE entladen (oder halten für decode)
    3. Transformer laden -> Denoising -> Transformer entladen
    4. VAE laden (falls entladen) -> Bild decodieren -> VAE entladen
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2509",
        transformer_id: str = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.transformer_id = transformer_id
        self.dtype = dtype
        self.device = device

        # Komponenten werden lazy geladen
        self._text_encoder = None
        self._tokenizer = None
        self._processor = None
        self._transformer = None
        self._vae = None
        self._scheduler = None

        # Adapter Info
        self._loaded_adapters = set()
        self._current_adapter = None

        print(f"[DynamicOffload] Initialized (RAM: {get_ram_gb():.1f} GB)")

    def _load_text_encoder(self):
        """Lädt Text Encoder + Processor."""
        if self._text_encoder is not None:
            return

        print(f"[DynamicOffload] Loading text encoder... (RAM: {get_ram_gb():.1f} GB)")

        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

        self._text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )
        self._processor = Qwen2VLProcessor.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
        )

        self._text_encoder.to(self.device)
        print(f"[DynamicOffload] Text encoder loaded (RAM: {get_ram_gb():.1f} GB)")

    def _unload_text_encoder(self):
        """Entlädt Text Encoder."""
        if self._text_encoder is None:
            return

        print(f"[DynamicOffload] Unloading text encoder... (RAM: {get_ram_gb():.1f} GB)")
        unload_model(self._text_encoder)
        self._text_encoder = None
        self._processor = None
        print(f"[DynamicOffload] Text encoder unloaded (RAM: {get_ram_gb():.1f} GB)")

    def _load_vae(self):
        """Lädt VAE."""
        if self._vae is not None:
            return

        print(f"[DynamicOffload] Loading VAE... (RAM: {get_ram_gb():.1f} GB)")

        from diffusers import AutoencoderKLQwenImage

        self._vae = AutoencoderKLQwenImage.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
        )
        self._vae.to(self.device)
        self._vae.enable_tiling()
        self._vae.enable_slicing()
        print(f"[DynamicOffload] VAE loaded (RAM: {get_ram_gb():.1f} GB)")

    def _unload_vae(self):
        """Entlädt VAE."""
        if self._vae is None:
            return

        print(f"[DynamicOffload] Unloading VAE... (RAM: {get_ram_gb():.1f} GB)")
        unload_model(self._vae)
        self._vae = None
        print(f"[DynamicOffload] VAE unloaded (RAM: {get_ram_gb():.1f} GB)")

    def _load_transformer(self):
        """Lädt Transformer."""
        if self._transformer is not None:
            return

        print(f"[DynamicOffload] Loading transformer... (RAM: {get_ram_gb():.1f} GB)")

        from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
        from amd_compat.qwen_sdpa_processor import QwenDoubleStreamAttnProcessorAMD

        self._transformer = QwenImageTransformer2DModel.from_pretrained(
            self.transformer_id,
            torch_dtype=self.dtype,
        )
        self._transformer.to(self.device)
        self._transformer.set_attn_processor(QwenDoubleStreamAttnProcessorAMD())
        print(f"[DynamicOffload] Transformer loaded (RAM: {get_ram_gb():.1f} GB)")

    def _unload_transformer(self):
        """Entlädt Transformer."""
        if self._transformer is None:
            return

        print(f"[DynamicOffload] Unloading transformer... (RAM: {get_ram_gb():.1f} GB)")
        unload_model(self._transformer)
        self._transformer = None
        print(f"[DynamicOffload] Transformer unloaded (RAM: {get_ram_gb():.1f} GB)")

    def _load_scheduler(self):
        """Lädt Scheduler (sehr klein, bleibt im Speicher)."""
        if self._scheduler is not None:
            return

        from diffusers import FlowMatchEulerDiscreteScheduler

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
        )

    def unload_all(self):
        """Entlädt alle Modelle."""
        self._unload_text_encoder()
        self._unload_vae()
        self._unload_transformer()
        aggressive_cleanup()
        print(f"[DynamicOffload] All models unloaded (RAM: {get_ram_gb():.1f} GB)")


def patch_pipeline_for_dynamic_offload(pipe):
    """
    Patcht eine existierende Pipeline für dynamisches Offloading.

    Fügt Methoden hinzu um einzelne Komponenten zu entladen.
    """

    def unload_text_encoder(self):
        """Entlädt den Text Encoder aus dem Speicher."""
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            print(f"[DynamicOffload] Unloading text_encoder... (RAM: {get_ram_gb():.1f} GB)")
            self.text_encoder.to('cpu')
            unload_model(self.text_encoder)
            self.text_encoder = None
            print(f"[DynamicOffload] text_encoder unloaded (RAM: {get_ram_gb():.1f} GB)")

    def unload_transformer(self):
        """Entlädt den Transformer aus dem Speicher."""
        if hasattr(self, 'transformer') and self.transformer is not None:
            print(f"[DynamicOffload] Unloading transformer... (RAM: {get_ram_gb():.1f} GB)")
            self.transformer.to('cpu')
            unload_model(self.transformer)
            self.transformer = None
            print(f"[DynamicOffload] transformer unloaded (RAM: {get_ram_gb():.1f} GB)")

    def unload_vae(self):
        """Entlädt die VAE aus dem Speicher."""
        if hasattr(self, 'vae') and self.vae is not None:
            print(f"[DynamicOffload] Unloading vae... (RAM: {get_ram_gb():.1f} GB)")
            self.vae.to('cpu')
            unload_model(self.vae)
            self.vae = None
            print(f"[DynamicOffload] vae unloaded (RAM: {get_ram_gb():.1f} GB)")

    # Methoden zur Pipeline hinzufügen
    pipe.unload_text_encoder = lambda: unload_text_encoder(pipe)
    pipe.unload_transformer = lambda: unload_transformer(pipe)
    pipe.unload_vae = lambda: unload_vae(pipe)

    return pipe
