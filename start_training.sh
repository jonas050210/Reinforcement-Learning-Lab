#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Training starten (Ubuntu)
# =============================================================================

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[FEHLER] .venv nicht gefunden. Bitte zuerst installieren:"
    echo "  ./install_ubuntu.sh"
    exit 1
fi

source .venv/bin/activate

echo "=============================================="
echo "  RL Training — Roboterarm (V1)"
echo "=============================================="
echo ""

# GPU prüfen
python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>/dev/null || echo "CPU-Modus"

echo ""
echo "Starte Training (500.000 Timesteps)..."
echo "Drücke Strg+C zum Abbrechen."
echo ""

python train.py --timesteps 500000 --device auto --curriculum 0
