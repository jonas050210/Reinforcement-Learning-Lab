#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Training starten
# =============================================================================
# Führe dieses Skript auf deinem PC aus, um das V1-Modell zu trainieren.
# 
# Voraussetzung: Virtualenv existiert (.venv)
# Dauer: ca. 5-6 Minuten auf i7-12700F (CPU)
# =============================================================================

cd "$(dirname "$0")"

echo "=============================================="
echo "  Reinforcement-Learning-Lab — Training V1"
echo "=============================================="
echo ""

# Virtualenv aktivieren
if [ ! -d ".venv" ]; then
    echo "[FEHLER] .venv nicht gefunden. Bitte zuerst einrichten:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install mujoco gymnasium stable-baselines3 torch tensorboard tqdm rich"
    exit 1
fi

source .venv/bin/activate

echo ">>> Starte PPO-Training (500.000 Timesteps, 4 parallele Umgebungen)"
echo ">>> Das kann 5-10 Minuten dauern..."
echo ""

python train.py

echo ""
echo "=============================================="
echo "  Training abgeschlossen!"
echo "=============================================="
echo ""
echo "Modell gespeichert in: ./ppo_arm_logs_final/ppo_wheel_arm_final_v2"
echo ""
echo "Jetzt testen:"
echo "  ./start_eval.sh"
echo "  ./start_demo.sh"
echo ""
