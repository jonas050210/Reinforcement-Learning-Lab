# 🤖 Reinforcement-Learning-Lab — Ubuntu Ready

Ein fertiges Reinforcement-Learning-Projekt für Linux Ubuntu mit 3D-Roboter, GPU-Unterstützung und Desktop-GUI.

## 🖥️ Für Ubuntu gebaut

Dieses Projekt läuft nativ auf Ubuntu/Debian mit echter GUI (kein reines Terminal).

| Komponente | Ubuntu | GPU (RTX 4060 Ti) |
|------------|--------|-------------------|
| V1: Roboterarm | ✅ Ja | ✅ CUDA beschleunigt |
| V2: 3 Werkzeuge | ✅ Ja | ✅ CUDA beschleunigt |
| V3: Roboterhund | 📋 Geplant | ✅ Isaac Lab Ready |
| Desktop-GUI | ✅ Ja | ✅ Echtes Fenster |

---

## 🚀 Schnellstart auf Ubuntu

### 1. Installation (einmalig)

```bash
cd ~/Reinforcement-Learning-Lab
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

Das installiert Python, MuJoCo, Stable Baselines3, CUDA-Treiber und erstellt einen Desktop-Shortcut.

### 2. Training starten

```bash
./start_training.sh
```

Oder doppelklicke auf den Desktop-Shortcut "RL Training".

### 3. Live-Demo ansehen (mit Fenster)

```bash
./start_demo.sh
```

Öffnet ein echtes MuJoCo-Fenster mit dem trainierten Arm.

---

## 🧠 Was trainiert wird

### V1: Roboterarm auf Rädern (fertig)
- 3-Achsen-Arm mit fahrbarer Basis
- Aufgabe: Würfel greifen und in Zielbox legen
- Algorithmus: PPO (Proximal Policy Optimization)
- Ergebnis: ~22% Erfolgsrate, 100% Heberate

### V2: Arm mit 3 Werkzeugen (optimiert)
- Parallel-Greifer, Sauger/Magnet, Schaufel
- 3 verschiedene Objekte mit passenden Zielboxen
- Automatisches Werkzeug-Wechseln
- Automatisches Curriculum (wird schwerer mit der Zeit)

### V3: Roboterhund (geplant)
- Unitree A1/Go1 mit Isaac Lab
- Laufen lernen auf deiner RTX 4060 Ti

---

## 🎮 Desktop-GUI (wirkliches Fenster, nicht nur Terminal)

Nach der Installation erscheint auf deinem Ubuntu-Desktop:

- **RL Training** — startet das PPO-Training
- **RL Demo** — öffnet das MuJoCo-Fenster mit dem trainierten Arm
- **RL Eval** — auswertet das Modell mit Statistiken

Alle Programme öffnen echte GUI-Fenster (X11/Wayland), nicht nur Terminal.

---

## 🖥️ Hardware (getestet/empfohlen)

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| CPU | 4 Kerne | 8+ Kerne (i7-12700F ✅) |
| RAM | 8 GB | 16–32 GB (32 GB ✅) |
| GPU | Nicht nötig (CPU) | NVIDIA RTX 3070+ (RTX 4060 Ti ✅) |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 LTS |
| Speicher | 20 GB | 100 GB (Isaac Lab für V3) |

---

## 📁 Projektstruktur

```
Reinforcement-Learning-Lab/
├── install_ubuntu.sh      # Ubuntu-Installer (erstellt Shortcuts, installiert Pakete)
├── start_training.sh      # Training starten (mit GUI-Fenster-Option)
├── start_eval.sh          # Modell auswerten
├── start_demo.sh          # Live-Demo mit echtem Fenster
├── rl_desktop.desktop     # Ubuntu-Desktop-Shortcut
├── mujoco_arm.xml         # V1 3D-Welt (Arm, Räder, Würfel, Box)
├── mujoco_arm_v2.xml      # V2 3D-Welt (3 Werkzeuge, 3 Objekte)
├── arm_env.py             # V1 Gymnasium-Environment (GPU-ready)
├── arm_env_v2.py          # V2 Gymnasium-Environment (GPU-ready)
├── train.py               # V1 PPO-Training (CUDA-aware)
├── train_v2.py            # V2 PPO-Training (CUDA-aware)
├── gui_launcher.py        # Echter GUI-Launcher für Ubuntu
├── README.md              # Diese Datei
├── PROJEKT_PLAN_3D_ROBOTER_RL.md  # Master-Plan
└── .venv/                 # Python-Virtualenv
```

---

## 🔧 GPU-Unterstützung (RTX 4060 Ti)

Das Projekt erkennt automatisch deine NVIDIA-GPU:

- **MuJoCo GPU**: Wird verwendet wenn `mujoco` mit CUDA-Unterstützung installiert ist
- **PyTorch CUDA**: Wird automatisch genutzt wenn verfügbar (`device="cuda"`)
- **Isaac Lab Ready**: V3-Vorbereitung für Isaac Lab mit RTX-GPU

Prüfe GPU-Status:
```bash
source .venv/bin/activate
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"
```

---

## 📄 Lizenz

Privates Lernprojekt. MuJoCo: Apache 2.0. Stable Baselines3: MIT.
