#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Modell evaluieren
# =============================================================================

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[FEHLER] .venv nicht gefunden."
    exit 1
fi

source .venv/bin/activate

echo "=============================================="
echo "  Reinforcement-Learning-Lab — Evaluation V1"
echo "=============================================="
echo ""

python eval.py --model ./ppo_arm_logs_final/ppo_wheel_arm_final_v2 --episodes 50
