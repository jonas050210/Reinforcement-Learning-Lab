#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Live-Demo (mit Fenster)
# =============================================================================

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[FEHLER] .venv nicht gefunden."
    exit 1
fi

source .venv/bin/activate

echo "=============================================="
echo "  Reinforcement-Learning-Lab — Live-Demo V1"
echo "=============================================="
echo ""
echo "Ein Fenster öffnet sich und zeigt den trainierten Arm."
echo "Drücke Strg+C zum Beenden."
echo ""

python demo.py --model ./ppo_arm_logs_final/ppo_wheel_arm_final_v2 --episodes 3 --delay 0.02
