#!/bin/bash
# Startet GPU Logging im Hintergrund

cd /home/user/Dokumente/Projekte/AmdGpudrive

echo "Starte GPU Logger im Hintergrund..."
nohup python3 simple_gpu_logger.py > gpu_logger.log 2>&1 &

PID=$!
echo $PID > gpu_logger.pid

echo "✓ GPU Logger läuft (PID: $PID)"
echo "✓ Log: gpu_logger.log"
echo "✓ Stoppen: kill \$(cat gpu_logger.pid)"
