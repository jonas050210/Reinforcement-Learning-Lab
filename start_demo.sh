#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Live-Demo (Ubuntu, echtes Fenster)
# =============================================================================

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[FEHLER] .venv nicht gefunden."
    exit 1
fi

source .venv/bin/activate

echo "=============================================="
echo "  RL Demo — Live-Visualisierung"
echo "=============================================="
echo ""
echo "Ein MuJoCo-Fenster öffnet sich mit dem trainierten Arm."
echo "Fenster schließen zum Beenden."
echo ""

python demo.py --model ./ppo_arm_logs_final/ppo_wheel_arm_final_v2 --episodes 3 --delay 0.02
