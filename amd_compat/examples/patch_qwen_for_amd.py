#!/usr/bin/env python3
"""
Beispiel: Qwen-Image-Edit auf AMD GPU ausführen

Dieses Script zeigt wie man das Qwen-Image-Edit Repo
ohne Flash Attention 3 auf AMD ROCm GPUs zum Laufen bringt.

Verwendung:
    1. Ins Qwen Repo wechseln
    2. Dieses Script ausführen
    3. app.py normal starten

Alternativ: app.py direkt patchen (siehe unten)
"""

import sys
import os

# AMD Compat Modul zum Path hinzufügen
AMD_COMPAT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AMD_COMPAT_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(AMD_COMPAT_PATH))


def patch_qwen_module():
    """
    Patcht das qwenimage Modul um SDPA statt FA3 zu verwenden.
    Muss VOR dem Import von qwenimage aufgerufen werden.
    """
    from amd_compat.qwen_sdpa_processor import QwenDoubleStreamAttnProcessorAMD

    # Monkey-patch: Ersetze den FA3 Processor
    class FakeFA3Module:
        QwenDoubleStreamAttnProcessorFA3 = QwenDoubleStreamAttnProcessorAMD

        @staticmethod
        def _ensure_fa3_available():
            pass  # Kein Check nötig

    sys.modules["qwenimage.qwen_fa3_processor"] = FakeFA3Module()
    print("✅ qwenimage.qwen_fa3_processor durch AMD SDPA ersetzt")


def create_patched_app():
    """
    Erstellt eine gepatchte Version von app.py im aktuellen Verzeichnis.
    """
    patch_code = '''
# === AMD GPU PATCH START ===
import sys
import os

# AMD Compat Modul
AMD_COMPAT = "/home/user/Dokumente/Projekte/AmdGpudrive"
if AMD_COMPAT not in sys.path:
    sys.path.insert(0, AMD_COMPAT)

from amd_compat.qwen_sdpa_processor import QwenDoubleStreamAttnProcessorAMD

# Fake FA3 Modul das den AMD Processor verwendet
class _FakeFA3:
    QwenDoubleStreamAttnProcessorFA3 = QwenDoubleStreamAttnProcessorAMD
    @staticmethod
    def _ensure_fa3_available():
        pass

sys.modules["qwenimage.qwen_fa3_processor"] = _FakeFA3()
print("AMD GPU: Flash Attention 3 durch SDPA ersetzt")
# === AMD GPU PATCH END ===

'''
    return patch_code


def patch_existing_app(app_path: str):
    """
    Fügt den AMD Patch am Anfang einer existierenden app.py ein.
    """
    with open(app_path, "r") as f:
        content = f.read()

    if "AMD GPU PATCH" in content:
        print(f"⚠️  {app_path} ist bereits gepatcht")
        return False

    patch = create_patched_app()
    patched_content = patch + content

    backup_path = app_path + ".backup"
    os.rename(app_path, backup_path)
    print(f"📁 Backup erstellt: {backup_path}")

    with open(app_path, "w") as f:
        f.write(patched_content)

    print(f"✅ {app_path} gepatcht für AMD GPU")
    return True


def main():
    print("="*60)
    print("🔧 Qwen-Image-Edit AMD GPU Patch")
    print("="*60)

    # Option 1: Patch im Speicher (für Import)
    if len(sys.argv) > 1 and sys.argv[1] == "--patch-file":
        if len(sys.argv) < 3:
            print("Verwendung: python patch_qwen_for_amd.py --patch-file /pfad/zu/app.py")
            sys.exit(1)
        patch_existing_app(sys.argv[2])

    # Option 2: Zeige Patch Code
    elif len(sys.argv) > 1 and sys.argv[1] == "--show-patch":
        print("\nFüge diesen Code am Anfang deiner app.py ein:\n")
        print(create_patched_app())

    # Default: Anleitung
    else:
        print("""
Verwendung:

1. Patch direkt in app.py einfügen:
   python patch_qwen_for_amd.py --patch-file /pfad/zu/app.py

2. Patch Code anzeigen (manuell einfügen):
   python patch_qwen_for_amd.py --show-patch

3. Als Modul importieren:
   import patch_qwen_for_amd
   patch_qwen_for_amd.patch_qwen_module()
   # Dann normal qwenimage importieren

Was macht der Patch?
- Ersetzt Flash Attention 3 durch PyTorch SDPA
- SDPA funktioniert auf AMD ROCm GPUs
- Gleiche Funktionalität, kein NVIDIA Kernel nötig
""")


if __name__ == "__main__":
    main()
