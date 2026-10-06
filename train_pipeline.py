#!/usr/bin/env python3
"""
Training Pipeline: Wartet auf v7, startet dann v8
"""
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/home/user/Dokumente/Projekte/AmdGpudrive")
LOG_FILE = BASE_DIR / "pipeline.log"
V7_LOG = BASE_DIR / "training_v7.log"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def is_v7_done():
    if not V7_LOG.exists():
        return False
    content = V7_LOG.read_text()
    return "Training complete" in content

def main():
    os.chdir(BASE_DIR)

    log("=== Training Pipeline gestartet ===")
    log("Warte auf v7...")

    # Warte auf v7
    while not is_v7_done():
        time.sleep(60)
        # Zeige Fortschritt
        if V7_LOG.exists():
            lines = V7_LOG.read_text().split("\n")
            epochs = [l for l in lines if "Epoch" in l and "/500" in l]
            if epochs:
                log(f"v7 Fortschritt: {epochs[-1].strip()[:50]}")

    log("v7 fertig! Starte v8 mit 1000 Epochen...")

    # Starte v8
    cmd = [
        "python3", "cuda_translator/train_translator.py",
        "--data", "cuda_translator/training_data_big.json",
        "--epochs", "1000",
        "--batch-size", "16",
        "--lr", "0.0005",
        "--device", "cuda",
        "--model-out", "cuda_translator/translator_v8_1000ep.pt",
        "--eval-interval", "100"
    ]

    v8_log = BASE_DIR / "training_v8.log"
    with open(v8_log, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            env={**os.environ, "PATH": str(BASE_DIR / ".venv-rocm6.4/bin") + ":" + os.environ.get("PATH", "")}
        )

    log(f"v8 beendet mit Code {result.returncode}")
    log("=== Pipeline komplett! ===")

if __name__ == "__main__":
    main()
