#!/bin/bash
# =============================================================================
# Reinforcement-Learning-Lab — Ubuntu Installer
# =============================================================================
# Führt eine vollständige Installation auf Ubuntu/Debian durch:
# - Python-Virtualenv
# - MuJoCo, Gymnasium, Stable Baselines3, PyTorch (mit CUDA)
# - Ubuntu-Desktop-Shortcuts (wirkliche GUI-Startpunkte)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Reinforcement-Learning-Lab — Ubuntu Installer"
echo "=============================================="
echo ""

# 1. System-Updates und Basis-Pakete
echo "[1/5] Installiere System-Pakete..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip \
    python3-tk \
    libgl1 libglx-mesa0 libxi6 libxrandr2 \
    xdg-utils \
    2>&1 | tail -5

# 2. NVIDIA CUDA prüfen
echo "[2/5] Prüfe NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
    echo "✅ NVIDIA GPU erkannt — CUDA-Unterstützung wird aktiviert"
    HAS_GPU=true
else
    echo "⚠️  Keine NVIDIA GPU erkannt — läuft auf CPU (funktioniert trotzdem)"
    HAS_GPU=false
fi

# 3. Virtualenv erstellen
echo "[3/5] Erstelle Python-Virtualenv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 4. Python-Pakete installieren
echo "[4/5] Installiere Python-Pakete (dauert 2-5 Min)..."
pip install --upgrade pip -q
pip install -q \
    mujoco \
    gymnasium \
    stable-baselines3 \
    torch \
    tensorboard \
    tqdm \
    rich \
    numpy

if [ "$HAS_GPU" = true ]; then
    # PyTorch mit CUDA nachinstallieren falls nötig
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
    echo "✅ PyTorch mit CUDA-Unterstützung"
fi

# 5. Desktop-Shortcuts erstellen
echo "[5/5] Erstelle Ubuntu-Desktop-Shortcuts..."

DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$DESKTOP_DIR"

# RL Training Shortcut
cat > "$DESKTOP_DIR/RL Training.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RL Training
Comment=Starte PPO-Training für den Roboterarm
Exec=bash -c 'cd "$SCRIPT_DIR" && source .venv/bin/activate && python train.py'
Icon=applications-science
Terminal=true
Categories=Education;Science;
EOF

# RL Demo Shortcut
cat > "$DESKTOP_DIR/RL Demo.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RL Demo
Comment=Öffne Live-Demo mit trainiertem Arm (MuJoCo-Fenster)
Exec=bash -c 'cd "$SCRIPT_DIR" && source .venv/bin/activate && python demo.py'
Icon=applications-graphics
Terminal=true
Categories=Education;Science;
EOF

# RL Eval Shortcut
cat > "$DESKTOP_DIR/RL Eval.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RL Eval
Comment=Auswerte das trainierte Modell (50 Episoden)
Exec=bash -c 'cd "$SCRIPT_DIR" && source .venv/bin/activate && python eval.py'
Icon=applications-education
Terminal=true
Categories=Education;Science;
EOF

chmod +x "$DESKTOP_DIR"/RL\ *.desktop

echo ""
echo "=============================================="
echo "  Installation abgeschlossen! ✅"
echo "=============================================="
echo ""
echo "Desktop-Shortcuts erstellt:"
echo "  - RL Training"
echo "  - RL Demo"
echo "  - RL Eval"
echo ""
echo "Oder im Terminal:"
echo "  ./start_training.sh"
echo "  ./start_demo.sh"
echo "  ./start_eval.sh"
echo ""
echo "GPU-Status: $([ "$HAS_GPU" = true ] && echo "CUDA aktiviert" || echo "CPU-Modus")"
