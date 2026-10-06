"""
Baut ein Standalone-Executable für Kollegen ohne Python.

Nutzung:
    pip install pyinstaller
    python build_standalone.py

Output: dist/amd_collector.exe
"""

import subprocess
import sys
import os
from pathlib import Path

def build():
    # Check pyinstaller
    try:
        import PyInstaller
    except ImportError:
        print("Installiere PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Build command
    script_dir = Path(__file__).parent
    main_script = script_dir / "standalone_collector.py"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "amd_collector",
        "--add-data", f"{script_dir / 'KOLLEGEN_ANLEITUNG.md'};.",
        str(main_script)
    ]

    print(f"Building: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=script_dir.parent)

    print("""
╔══════════════════════════════════════════════════════════════╗
║                    BUILD FERTIG!                             ║
╠══════════════════════════════════════════════════════════════╣
║  Output: dist/amd_collector.exe                              ║
║                                                              ║
║  Das kann dein Kollege einfach starten - kein Python nötig!  ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    build()
