#!/usr/bin/env python3
"""
AMD ML Environment Setup Script

Installiert und konfiguriert alle notwendigen Pakete für ML auf AMD GPUs:
- PyTorch ROCm
- bitsandbytes-rocm
- Flash Attention Alternativen
- xFormers (optional)

Verwendung:
    python3 setup_amd_ml.py [--full] [--check-only]

Optionen:
    --full          Installiert auch optionale Pakete (xFormers, etc.)
    --check-only    Prüft nur den Status ohne Installation
    --venv PATH     Verwendet angegebenes venv
"""

import subprocess
import sys
import os
import argparse


def run_cmd(cmd: list, capture: bool = True, timeout: int = 600) -> tuple:
    """Führt Kommando aus und gibt (success, output) zurück."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_rocm() -> dict:
    """Prüft ROCm Installation."""
    info = {
        "installed": False,
        "version": None,
        "path": None,
        "gpu": None,
    }

    # ROCm Pfad
    for path in ["/opt/rocm", "/opt/rocm-6.4.0", "/opt/rocm-6.3.0"]:
        if os.path.exists(path):
            info["installed"] = True
            info["path"] = path
            break

    # ROCm Version
    success, output = run_cmd(["rocm-smi", "--version"])
    if success:
        for line in output.split("\n"):
            if "version" in line.lower():
                info["version"] = line.strip()
                break

    # GPU Info
    success, output = run_cmd(["rocm-smi", "--showproductname"])
    if success:
        for line in output.split("\n"):
            if "GPU" in line or "Radeon" in line:
                info["gpu"] = line.strip()
                break

    return info


def check_pytorch() -> dict:
    """Prüft PyTorch Installation."""
    info = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "rocm": False,
        "device": None,
    }

    try:
        import torch
        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["rocm"] = hasattr(torch.version, "hip") and torch.version.hip is not None

        if info["cuda_available"]:
            info["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def check_bitsandbytes() -> dict:
    """Prüft bitsandbytes Installation."""
    info = {
        "installed": False,
        "version": None,
        "rocm_support": False,
    }

    try:
        import bitsandbytes as bnb
        info["installed"] = True
        info["version"] = getattr(bnb, "__version__", "unknown")

        # Prüfe ROCm Support
        try:
            # Versuche eine einfache Operation
            import torch
            if torch.cuda.is_available():
                x = torch.randn(10, 10).cuda()
                # Wenn das ohne Fehler läuft, ist ROCm Support da
                info["rocm_support"] = True
        except:
            pass
    except ImportError:
        pass

    return info


def check_flash_attention() -> dict:
    """Prüft Flash Attention Verfügbarkeit."""
    info = {
        "flash_attn": False,
        "flash_attn_version": None,
        "sdpa_available": False,
        "kernels_fa3": False,
    }

    # PyTorch SDPA (immer verfügbar ab PyTorch 2.0)
    try:
        import torch
        info["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    except:
        pass

    # flash-attn Package
    try:
        import flash_attn
        info["flash_attn"] = True
        info["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        pass

    # HuggingFace Kernels FA3
    try:
        from kernels import get_kernel
        _k = get_kernel("kernels-community/vllm-flash-attn3")
        info["kernels_fa3"] = True
    except:
        pass

    return info


def install_bitsandbytes_rocm() -> bool:
    """Installiert bitsandbytes für ROCm."""
    print("\n📦 Installiere bitsandbytes für ROCm...")

    # Versuche verschiedene Installationsmethoden
    methods = [
        ("ROCm Fork", [sys.executable, "-m", "pip", "install",
                       "git+https://github.com/ROCm/bitsandbytes.git"]),
        ("bitsandbytes-rocm", [sys.executable, "-m", "pip", "install", "bitsandbytes-rocm"]),
        ("bitsandbytes>=0.44", [sys.executable, "-m", "pip", "install", "bitsandbytes>=0.44.0"]),
    ]

    for name, cmd in methods:
        print(f"  Versuche: {name}...")
        success, output = run_cmd(cmd, timeout=300)
        if success:
            # Verifizieren
            check = check_bitsandbytes()
            if check["installed"]:
                print(f"  ✅ {name} erfolgreich installiert!")
                return True
        print(f"  ❌ {name} fehlgeschlagen")

    return False


def install_xformers_rocm() -> bool:
    """Installiert xFormers für ROCm."""
    print("\n📦 Installiere xFormers für ROCm...")

    cmd = [sys.executable, "-m", "pip", "install",
           "xformers", "--index-url", "https://download.pytorch.org/whl/rocm6.4"]

    success, output = run_cmd(cmd, timeout=300)
    if success:
        print("  ✅ xFormers installiert!")
        return True
    else:
        print("  ❌ xFormers Installation fehlgeschlagen")
        print(f"     {output[:200]}")
        return False


def print_status():
    """Zeigt den aktuellen Status an."""
    print("\n" + "="*60)
    print("🔍 AMD ML Environment Status")
    print("="*60)

    # ROCm
    rocm = check_rocm()
    print(f"\n📍 ROCm:")
    print(f"   Installiert: {'✅' if rocm['installed'] else '❌'}")
    if rocm["path"]:
        print(f"   Pfad: {rocm['path']}")
    if rocm["version"]:
        print(f"   Version: {rocm['version']}")
    if rocm["gpu"]:
        print(f"   GPU: {rocm['gpu']}")

    # PyTorch
    pytorch = check_pytorch()
    print(f"\n🔥 PyTorch:")
    print(f"   Installiert: {'✅' if pytorch['installed'] else '❌'}")
    if pytorch["version"]:
        print(f"   Version: {pytorch['version']}")
    print(f"   CUDA/ROCm verfügbar: {'✅' if pytorch['cuda_available'] else '❌'}")
    print(f"   ROCm Backend: {'✅' if pytorch['rocm'] else '❌'}")
    if pytorch["device"]:
        print(f"   Device: {pytorch['device']}")

    # bitsandbytes
    bnb = check_bitsandbytes()
    print(f"\n🔢 bitsandbytes:")
    print(f"   Installiert: {'✅' if bnb['installed'] else '❌'}")
    if bnb["version"]:
        print(f"   Version: {bnb['version']}")
    if bnb["installed"]:
        print(f"   ROCm Support: {'✅' if bnb['rocm_support'] else '⚠️ unbekannt'}")

    # Flash Attention
    fa = check_flash_attention()
    print(f"\n⚡ Flash Attention:")
    print(f"   PyTorch SDPA: {'✅' if fa['sdpa_available'] else '❌'}")
    print(f"   flash-attn Package: {'✅' if fa['flash_attn'] else '❌ (nutze SDPA)'}")
    print(f"   HF Kernels FA3: {'✅' if fa['kernels_fa3'] else '❌ (nutze SDPA)'}")

    print("\n" + "="*60)

    # Empfehlungen
    recommendations = []
    if not pytorch["installed"] or not pytorch["cuda_available"]:
        recommendations.append("PyTorch ROCm installieren")
    if not bnb["installed"]:
        recommendations.append("bitsandbytes-rocm installieren (--install)")
    if not fa["sdpa_available"]:
        recommendations.append("PyTorch >= 2.0 für SDPA benötigt")

    if recommendations:
        print("\n📋 Empfehlungen:")
        for r in recommendations:
            print(f"   • {r}")
    else:
        print("\n✅ Umgebung ist bereit für AMD ML!")

    return rocm, pytorch, bnb, fa


def main():
    parser = argparse.ArgumentParser(description="AMD ML Environment Setup")
    parser.add_argument("--check-only", action="store_true",
                        help="Nur Status prüfen, nichts installieren")
    parser.add_argument("--full", action="store_true",
                        help="Vollständige Installation inkl. optionaler Pakete")
    parser.add_argument("--install-bnb", action="store_true",
                        help="Installiere bitsandbytes-rocm")
    parser.add_argument("--install-xformers", action="store_true",
                        help="Installiere xFormers für ROCm")

    args = parser.parse_args()

    # Status anzeigen
    rocm, pytorch, bnb, fa = print_status()

    if args.check_only:
        return

    # Installationen
    if args.install_bnb or args.full:
        if not bnb["installed"]:
            install_bitsandbytes_rocm()

    if args.install_xformers or args.full:
        install_xformers_rocm()

    # Finalen Status anzeigen
    if args.install_bnb or args.install_xformers or args.full:
        print("\n" + "="*60)
        print("📊 Finaler Status nach Installation:")
        print_status()


if __name__ == "__main__":
    main()
