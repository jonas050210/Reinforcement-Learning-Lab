# 🤖 Reinforcement-Learning-Lab

Ein Lab für Reinforcement-Learning-Experimente mit 3D-Robotern in der Simulation.

## 📦 Aktueller Stand: Version 1 – Roboterarm auf Rädern

**Aufgabe:** Ein 3-Achsen-Roboterarm auf fahrbarer Basis lernt mit PPO (Proximal Policy Optimization), einen Würfel zu greifen und in eine Zielbox zu legen.

**Ergebnis (nach Training):**
| Metrik | Wert |
|--------|------|
| Erfolgsrate (Würfel in Box) | ~22% |
| Würfel zuverlässig heben | 100% |
| Training | 200k Timesteps auf CPU |
| Simulator | MuJoCo (kostenlos, läuft auf CPU) |

## 🗂️ Projektstruktur

```
Reinforcement-Learning-Lab/
├── mujoco_arm.xml          # MuJoCo-Welt: Arm, Räder, Würfel, Box
├── arm_env.py              # Gymnasium Environment (RL-Umgebung)
├── train.py                # PPO-Trainingsskript
├── eval.py                 # Evaluierung eines trainierten Modells
├── demo.py                 # Live-Demo mit Visualisierung
├── PROJEKT_PLAN_3D_ROBOTER_RL.md  # Ausführlicher Plan (alle 3 Versionen)
├── ppo_arm_logs_final/     # Trainierte Modelle & Logs
│   ├── ppo_wheel_arm_best.zip      # Bestes Modell (22% Erfolgsrate)
│   ├── ppo_wheel_arm_final_v2.zip  # Letztes Modell (500k Timesteps)
│   └── best_model_v5.zip           # Backup vom v5-Training
├── .venv/                  # Python-Virtualenv (nicht ins Git)
└── .gitignore              # Git-Ignore Regeln
```

## 🚀 Schnellstart

### 1. Umgebung vorbereiten

```bash
cd /home/user/Reinforcement-Learning-Lab

# Virtualenv erstellen (nur einmalig)
python3 -m venv .venv
source .venv/bin/activate

# Pakete installieren
pip install mujoco gymnasium stable-baselines3 torch tensorboard tqdm rich
```

### 2. Training starten (neu trainieren)

```bash
source .venv/bin/activate
python train.py
```

Das trainiert PPO mit 4 parallelen Umgebungen für 500k Timesteps. Dauert ca. 5–6 Minuten auf einem i7-12700F.

### 3. Modell evaluieren

```bash
source .venv/bin/activate
python eval.py --model ./ppo_arm_logs_final/ppo_wheel_arm_best.zip --episodes 50
```

### 4. Live-Demo ansehen (mit Fenster)

```bash
source .venv/bin/activate
python demo.py --model ./ppo_arm_logs_final/ppo_wheel_arm_best.zip --episodes 3
```

## 🧠 Wie es funktioniert

- **Simulator:** MuJoCo (Physik-Engine für Robotik, kostenlos, CPU-fähig)
- **RL-Algorithmus:** PPO über Stable Baselines3
- **Observation:** Gelenkpositionen, -geschwindigkeiten, Würfel-Position, Greifer-Position, Relativvektoren (Greifer→Würfel, Würfel→Box)
- **Action:** Relative Änderungen der 5 Antriebe (Basis X, Basis Y, Gelenk 1–3)
- **Belohnung (Reward Shaping):**
  - Annäherung an Würfel: +5 × Distanz-Verbesserung
  - Würfel zur Box bewegen: +10 × Distanz-Verbesserung
  - Greifer berührt Würfel: +2
  - Würfel heben (einmalig): +5
  - Würfel in Box: +100
  - Zeitstrafe pro Schritt: −0.1

## 🛠️ Geplante Versionen (siehe PROJEKT_PLAN_3D_ROBOTER_RL.md)

| Version | Roboter | Aufgabe | Simulator | Status |
|---------|---------|---------|-----------|--------|
| V1 | 3-Achsen-Arm auf Rädern + 3 Werkzeuge | Greifen & in Box legen | MuJoCo | ✅ Funktioniert |
| V2 | Roboterhund (Unitree A1/Go1) | Laufen lernen | Isaac Lab (braucht RTX) | 📋 Geplant |
| V3 | 3D-Mensch / Humanoid | Aufrecht stehen & gehen | Isaac Lab | 📋 Geplant |

## 🖥️ Hardware (getestet)

- CPU: Intel i7-12700F
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 4060 Ti 8 GB (für V1 nicht nötig, MuJoCo läuft auf CPU)
- OS: Linux (Ubuntu/Debian)

## 🐛 Bekannte Einschränkungen

- Die Erfolgsrate liegt bei ~22%. Für eine robuste Lösung braucht es mehr Training oder Curriculum-Lernstufen (Würfel startet immer weiter weg).
- Die 3 austauschbaren Werkzeuge (Greifer, Sauger, Schaufel) sind im Plan vorgesehen, noch nicht im Code umgesetzt.
- Version 2 (Hund) erfordert NVIDIA Isaac Lab + Isaac Sim, was eine RTX-GPU und ~50 GB Speicher benötigt.

## 📄 Lizenz

Privates Lernprojekt. Alle verwendeten 3D-Modelle und Simulatoren unterliegen ihren jeweiligen Lizenzen (MuJoCo: Apache 2.0, Stable Baselines3: MIT).
