"""
AMD GPU Memory Guard - Verhindert System-Freezes bei OOM

Dieses Modul bietet:
1. Echtzeit RAM + VRAM Monitoring
2. Automatischen Abbruch bei kritischen Speicherwerten
3. Aggressives Memory Cleanup zwischen Steps

Verwendung:
    from amd_compat.memory_guard import MemoryGuard, get_memory_status

    # Standalone Status
    print(get_memory_status())

    # Als Context Manager
    with MemoryGuard(vram_limit_gb=20, ram_limit_gb=55):
        # Dein ML Code hier
        result = pipe(...)

    # Als Callback für Diffusion Pipelines
    guard = MemoryGuard(vram_limit_gb=20)
    result = pipe(..., callback_on_step_end=guard.step_callback)
"""

import gc
import os
import subprocess
import threading
import time
from typing import Callable, Optional, Dict, Any

import torch


class MemoryLimitExceeded(Exception):
    """Wird geworfen wenn Speicherlimits überschritten werden."""
    pass


def get_vram_usage_gb() -> tuple[float, float]:
    """
    Gibt (benutzt_gb, total_gb) für AMD GPU zurück.
    Verwendet rocm-smi für genaue Werte.
    """
    try:
        result = subprocess.run(
            ['/opt/rocm/bin/rocm-smi', '--showmeminfo', 'vram', '--json'],
            capture_output=True, text=True, timeout=5
        )
        import json
        data = json.loads(result.stdout)

        # ROCm 6.x JSON Format
        if 'card0' in data:
            card = data['card0']
            used = int(card.get('VRAM Total Used Memory (B)', 0))
            total = int(card.get('VRAM Total Memory (B)', 1))
        else:
            # Alternatives Format
            for key, val in data.items():
                if 'GPU' in key or 'card' in key.lower():
                    used = int(val.get('VRAM Total Used Memory (B)', 0))
                    total = int(val.get('VRAM Total Memory (B)', 1))
                    break
            else:
                # Fallback auf PyTorch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                else:
                    return 0.0, 0.0

        return used / (1024**3), total / (1024**3)
    except Exception:
        # Fallback auf PyTorch memory stats
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return used / (1024**3), total / (1024**3)
        return 0.0, 0.0


def get_ram_usage_gb() -> tuple[float, float]:
    """
    Gibt (benutzt_gb, total_gb) für System RAM zurück.
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()

        total = available = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total = int(line.split()[1]) * 1024  # kB to B
            elif line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024

        used = total - available
        return used / (1024**3), total / (1024**3)
    except Exception:
        return 0.0, 0.0


def get_memory_status() -> str:
    """
    Gibt einen formatierten String mit aktuellem Speicherstatus zurück.
    """
    vram_used, vram_total = get_vram_usage_gb()
    ram_used, ram_total = get_ram_usage_gb()

    vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
    ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0

    return (
        f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB ({vram_pct:.0f}%) | "
        f"RAM: {ram_used:.1f}/{ram_total:.1f} GB ({ram_pct:.0f}%)"
    )


def aggressive_cleanup():
    """
    Aggressives Memory Cleanup - ruft gc und CUDA cache leer.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryGuard:
    """
    Memory Guard der Speicherverbrauch überwacht und bei kritischen
    Werten abbricht um System-Freezes zu verhindern.

    Args:
        vram_limit_gb: Maximaler VRAM in GB (default: 22 für 24GB GPUs)
        ram_limit_gb: Maximaler RAM in GB (default: 58 für 64GB Systeme)
        check_interval: Sekunden zwischen Checks im Background Thread
        cleanup_every_step: Ob nach jedem Step cleanup gemacht werden soll
        verbose: Ob Status-Meldungen gedruckt werden sollen
    """

    def __init__(
        self,
        vram_limit_gb: float = 22.0,
        ram_limit_gb: float = 58.0,
        check_interval: float = 1.0,
        cleanup_every_step: bool = True,
        verbose: bool = True
    ):
        self.vram_limit_gb = vram_limit_gb
        self.ram_limit_gb = ram_limit_gb
        self.check_interval = check_interval
        self.cleanup_every_step = cleanup_every_step
        self.verbose = verbose

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._exceeded = False
        self._step_count = 0

    def check_limits(self) -> bool:
        """
        Prüft ob Speicherlimits überschritten wurden.
        Returns True wenn alles OK, False wenn Limit überschritten.
        """
        vram_used, _ = get_vram_usage_gb()
        ram_used, _ = get_ram_usage_gb()

        if vram_used > self.vram_limit_gb:
            if self.verbose:
                print(f"\n[MemoryGuard] VRAM Limit überschritten: {vram_used:.1f} > {self.vram_limit_gb} GB")
            return False

        if ram_used > self.ram_limit_gb:
            if self.verbose:
                print(f"\n[MemoryGuard] RAM Limit überschritten: {ram_used:.1f} > {self.ram_limit_gb} GB")
            return False

        return True

    def _monitor_loop(self):
        """Background Thread der kontinuierlich Speicher überwacht."""
        while not self._stop_event.is_set():
            if not self.check_limits():
                self._exceeded = True
                aggressive_cleanup()
                break
            self._stop_event.wait(self.check_interval)

    def start_monitoring(self):
        """Startet Background Monitoring Thread."""
        self._exceeded = False
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        if self.verbose:
            print(f"[MemoryGuard] Monitoring gestartet (VRAM < {self.vram_limit_gb}GB, RAM < {self.ram_limit_gb}GB)")

    def stop_monitoring(self):
        """Stoppt Background Monitoring Thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if self.verbose:
            print("[MemoryGuard] Monitoring gestoppt")

    def step_callback(self, pipe, step: int, timestep: int, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callback für Diffusion Pipelines - wird nach jedem Denoising Step aufgerufen.

        Verwendung:
            result = pipe(..., callback_on_step_end=guard.step_callback)
        """
        self._step_count = step

        if self.cleanup_every_step:
            aggressive_cleanup()

        if not self.check_limits():
            self._exceeded = True
            raise MemoryLimitExceeded(
                f"Memory limit exceeded at step {step}. {get_memory_status()}"
            )

        if self.verbose and step % 5 == 0:
            print(f"  Step {step}: {get_memory_status()}")

        return callback_kwargs

    def __enter__(self):
        """Context Manager Entry - startet Monitoring."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit - stoppt Monitoring und cleanup."""
        self.stop_monitoring()
        aggressive_cleanup()

        if self._exceeded and exc_type is None:
            raise MemoryLimitExceeded(
                f"Memory limit exceeded during execution. {get_memory_status()}"
            )

        return False  # Exception nicht unterdrücken


def setup_memory_optimizations():
    """
    Setzt globale PyTorch Einstellungen für optimale Speichernutzung auf AMD.

    Sollte am Anfang des Scripts aufgerufen werden.
    """
    # Expandable Segments für bessere Fragmentierung
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ROCm spezifische Optimierungen
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    # CUDA Memory Fraction (lässt etwas Headroom)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)

    # Disable TF32 für konsistentere Speichernutzung
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("[MemoryGuard] Memory Optimierungen aktiviert")


def apply_aggressive_offload(pipe):
    """
    Wendet aggressive CPU Offload Einstellungen auf eine Pipeline an.

    Args:
        pipe: Eine diffusers Pipeline
    """
    # Sequential CPU Offload - nur ein Layer gleichzeitig auf GPU
    if hasattr(pipe, 'enable_sequential_cpu_offload'):
        pipe.enable_sequential_cpu_offload()
        print("[MemoryGuard] Sequential CPU Offload aktiviert")
    elif hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
        print("[MemoryGuard] Model CPU Offload aktiviert")

    # VAE Optimierungen
    if hasattr(pipe, 'vae'):
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
        print("[MemoryGuard] VAE Tiling + Slicing aktiviert")

    # Attention Slicing für Transformer
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing("auto")
        print("[MemoryGuard] Attention Slicing aktiviert")


if __name__ == "__main__":
    # Test
    print("Memory Guard Test")
    print("=" * 50)
    print(get_memory_status())
    print()

    print("Starte Monitoring für 5 Sekunden...")
    with MemoryGuard(vram_limit_gb=23, ram_limit_gb=60, verbose=True) as guard:
        for i in range(5):
            time.sleep(1)
            print(f"  Iteration {i+1}: {get_memory_status()}")

    print("\nTest abgeschlossen.")
