#!/bin/bash
# Swap-File Setup für Linux
# Verhindert System-Freezes bei OOM (Out of Memory)
#
# WICHTIG: Ohne Swap friert Linux bei vollem RAM komplett ein!
# Der OOM-Killer reagiert oft zu langsam, besonders bei GPU-Workloads.
#
# Verwendung:
#   sudo ./setup_swap.sh          # Erstellt 32GB Swap
#   sudo ./setup_swap.sh 64       # Erstellt 64GB Swap
#   sudo ./setup_swap.sh status   # Zeigt aktuellen Swap Status
#   sudo ./setup_swap.sh remove   # Entfernt Swap-File

SWAP_FILE="/swapfile"
DEFAULT_SIZE_GB=32

show_status() {
    echo "=== Swap Status ==="
    swapon --show
    echo ""
    echo "=== Memory Info ==="
    free -h
    echo ""
    echo "=== Swappiness ==="
    cat /proc/sys/vm/swappiness
}

create_swap() {
    local SIZE_GB=$1

    if [ -f "$SWAP_FILE" ]; then
        echo "Swap-File existiert bereits: $SWAP_FILE"
        echo "Verwende './setup_swap.sh remove' zum Entfernen."
        exit 1
    fi

    echo "Erstelle ${SIZE_GB}GB Swap-File..."

    # Swap-File erstellen
    fallocate -l ${SIZE_GB}G $SWAP_FILE
    if [ $? -ne 0 ]; then
        echo "fallocate fehlgeschlagen, versuche dd..."
        dd if=/dev/zero of=$SWAP_FILE bs=1G count=$SIZE_GB status=progress
    fi

    # Permissions setzen
    chmod 600 $SWAP_FILE

    # Als Swap formatieren
    mkswap $SWAP_FILE

    # Aktivieren
    swapon $SWAP_FILE

    # Swappiness auf 10 setzen (nur bei RAM-Knappheit swappen)
    echo 10 > /proc/sys/vm/swappiness

    echo ""
    echo "=== Swap aktiviert ==="
    show_status

    echo ""
    echo "HINWEIS: Für permanenten Swap nach Neustart, füge folgendes zu /etc/fstab hinzu:"
    echo "$SWAP_FILE none swap sw 0 0"
    echo ""
    echo "Und für permanente Swappiness zu /etc/sysctl.conf:"
    echo "vm.swappiness=10"
}

remove_swap() {
    if [ ! -f "$SWAP_FILE" ]; then
        echo "Kein Swap-File gefunden: $SWAP_FILE"
        exit 1
    fi

    echo "Deaktiviere Swap..."
    swapoff $SWAP_FILE

    echo "Lösche Swap-File..."
    rm $SWAP_FILE

    echo "Swap entfernt."
    show_status
}

# Argument verarbeiten
case "$1" in
    status)
        show_status
        ;;
    remove)
        if [ "$EUID" -ne 0 ]; then
            echo "Bitte als root ausführen: sudo $0 remove"
            exit 1
        fi
        remove_swap
        ;;
    ""|[0-9]*)
        if [ "$EUID" -ne 0 ]; then
            echo "Bitte als root ausführen: sudo $0 [size_gb]"
            exit 1
        fi
        SIZE=${1:-$DEFAULT_SIZE_GB}
        create_swap $SIZE
        ;;
    *)
        echo "Verwendung: $0 [status|remove|size_in_gb]"
        echo "  status  - Zeigt aktuellen Swap Status"
        echo "  remove  - Entfernt Swap-File"
        echo "  32      - Erstellt 32GB Swap (default)"
        exit 1
        ;;
esac
