"""
bitsandbytes AMD/ROCm Kompatibilität

bitsandbytes ist eine Library für 8-bit/4-bit Quantisierung die ursprünglich
nur NVIDIA GPUs unterstützte. Es gibt mehrere Optionen für AMD:

1. bitsandbytes >= 0.44.0 hat experimentellen ROCm Support
2. ROCm Fork: https://github.com/ROCm/bitsandbytes (empfohlen)
3. Fallback: Quantisierung deaktivieren und fp16/bf16 verwenden
"""

import subprocess
import sys
import os
from typing import Optional, Dict, Any


def is_bitsandbytes_available() -> bool:
    """Prüft ob bitsandbytes importiert werden kann."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


def get_bitsandbytes_version() -> Optional[str]:
    """Gibt die installierte bitsandbytes Version zurück."""
    try:
        import bitsandbytes
        return getattr(bitsandbytes, "__version__", "unknown")
    except ImportError:
        return None


def is_rocm_environment() -> bool:
    """Prüft ob wir in einer ROCm Umgebung sind."""
    # Prüfe ROCm Umgebungsvariablen
    if os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH"):
        return True

    # Prüfe ob rocm-smi verfügbar ist
    try:
        result = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

    # Prüfe PyTorch ROCm
    try:
        import torch
        return torch.version.hip is not None
    except:
        return False


def ensure_bitsandbytes_rocm(force_reinstall: bool = False) -> bool:
    """
    Stellt sicher dass bitsandbytes für ROCm installiert ist.

    Args:
        force_reinstall: Erzwingt Neuinstallation auch wenn bereits vorhanden

    Returns:
        True wenn erfolgreich, False sonst
    """
    if not is_rocm_environment():
        print("WARNUNG: Keine ROCm Umgebung erkannt!")
        return False

    if is_bitsandbytes_available() and not force_reinstall:
        version = get_bitsandbytes_version()
        print(f"bitsandbytes bereits installiert: {version}")
        return True

    print("Installiere bitsandbytes für ROCm...")

    # Versuche zuerst den offiziellen ROCm Fork
    install_commands = [
        # Option 1: ROCm Fork (empfohlen)
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/ROCm/bitsandbytes.git", "--quiet"],
        # Option 2: bitsandbytes-rocm package
        [sys.executable, "-m", "pip", "install",
         "bitsandbytes-rocm", "--quiet"],
        # Option 3: Neueste bitsandbytes (>=0.44 hat ROCm support)
        [sys.executable, "-m", "pip", "install",
         "bitsandbytes>=0.44.0", "--quiet"],
    ]

    for cmd in install_commands:
        try:
            print(f"Versuche: {' '.join(cmd[-2:])}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                # Prüfe ob es funktioniert
                try:
                    # Importiere in neuem Prozess um Cache-Probleme zu vermeiden
                    check = subprocess.run(
                        [sys.executable, "-c", "import bitsandbytes; print('OK')"],
                        capture_output=True, text=True, timeout=30
                    )
                    if "OK" in check.stdout:
                        print("bitsandbytes erfolgreich installiert!")
                        return True
                except:
                    pass
        except Exception as e:
            print(f"Fehlgeschlagen: {e}")
            continue

    print("WARNUNG: bitsandbytes konnte nicht installiert werden.")
    print("Quantisierung wird nicht verfügbar sein.")
    return False


def get_quantization_config(
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Erstellt eine Quantisierungs-Konfiguration für transformers/peft.

    Wenn bitsandbytes nicht verfügbar ist, wird None zurückgegeben
    und das Modell sollte ohne Quantisierung geladen werden.

    Args:
        load_in_8bit: 8-bit Quantisierung
        load_in_4bit: 4-bit Quantisierung
        bnb_4bit_compute_dtype: Compute dtype für 4-bit
        bnb_4bit_quant_type: Quantisierungstyp (nf4, fp4)
        bnb_4bit_use_double_quant: Double Quantisierung für mehr Speicherersparnis

    Returns:
        BitsAndBytesConfig dict oder None
    """
    if not is_bitsandbytes_available():
        print("WARNUNG: bitsandbytes nicht verfügbar - Quantisierung deaktiviert")
        return None

    if not (load_in_8bit or load_in_4bit):
        return None

    try:
        import torch
        from transformers import BitsAndBytesConfig

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=dtype_map.get(bnb_4bit_compute_dtype, torch.float16),
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        return config

    except Exception as e:
        print(f"WARNUNG: Quantisierungs-Config Fehler: {e}")
        return None


class BitsAndBytesUnavailableError(ImportError):
    """Wird geworfen wenn bitsandbytes benötigt aber nicht verfügbar ist."""

    def __init__(self, message: str = None):
        if message is None:
            message = (
                "bitsandbytes ist nicht verfügbar.\n"
                "Für AMD ROCm GPUs installiere:\n"
                "  pip install git+https://github.com/ROCm/bitsandbytes.git\n"
                "oder\n"
                "  pip install bitsandbytes-rocm\n"
                "\n"
                "Alternativ: Lade Modell ohne Quantisierung (mehr VRAM nötig)"
            )
        super().__init__(message)
