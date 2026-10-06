#!/bin/bash
# Stoppt GPU Logger

cd /home/user/Dokumente/Projekte/AmdGpudrive

if [ -f gpu_logger.pid ]; then
    PID=$(cat gpu_logger.pid)
    echo "Stoppe GPU Logger (PID: $PID)..."
    kill $PID 2>/dev/null
    rm gpu_logger.pid
    echo "✓ Gestoppt"
    echo "✓ Logs in: gpu_logs/"
else
    echo "Kein Logger läuft (gpu_logger.pid nicht gefunden)"
fi
