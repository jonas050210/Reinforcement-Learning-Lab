# 🤖 Reinforcement-Learning-Lab

Ein Lab für Reinforcement-Learning-Experimente mit 3D-Robotern in der Simulation.

---

## 📍 Wo läuft das? (auf deinem PC)

**Alles läuft lokal auf deinem Computer.** Du brauchst keine Cloud, keine API-Keys und kein Konto.

| Komponente | Wo | Was wird benötigt |
|------------|----|-------------------|
| Training | Dein PC (CPU) | Python 3.11, ca. 5-10 Min |
| Demo / Evaluation | Dein PC (CPU + Fenster) | MuJoCo-Viewer öffnet ein Fenster |
| 3D-Modelle | Im Repo (XML) | Kein externer Download nötig |

**Deine Hardware (i7-12700F, 32 GB DDR4, RTX 4060 Ti):**
- **V1 (Arm):** Läuft perfekt auf CPU, GPU nicht nötig
- **V2 (3 Werkzeuge):** Läuft auf CPU, GPU nicht nötig
- **V3 (Hund/Isaac Lab):** Würde die RTX 4060 Ti brauchen (geplant, nicht gestartet)

---

## 🚀 Schnellstart (3 Befehle)

```bash
# 1. In den Projektordner wechseln
cd /home/user/Reinforcement-Learning-Lab

# 2. Virtualenv aktivieren
source .venv/bin/activate

# 3. Training starten
./start_training.sh
```

Das war's. Das Skript installiert nichts, es trainiert nur.

---

## 🧠 Wie trainieren wir? (der Prozess)

```
1. Dein PC startet 4 parallele Simulationswelten (MuJoCo)
2. In jeder Welt versucht ein Roboterarm, einen Würfel zu greifen und in eine Box zu legen
3. Der Arm bekommt Belohnung für:
   - Näherkommen zum Würfel (+)
   - Würfel berühren (+)
   - Würfel heben (+)
   - Würfel in die Box legen (+++)
4. Die KI (PPO-Algorithmus) merkt sich, welche Aktionen zu mehr Belohnung führten
5. Nach 500.000 Versuchen (Timesteps) kann der Arm die Aufgabe zu ~20-30%
```

**Ergebnis nach Training:** Der Arm hebt den Würfel zuverlässig (100%) und bringt ihn in ca. jedem 4. Versuch in die Box.

---

## 🗂️ Projektstruktur

```
Reinforcement-Learning-Lab/
├── start_training.sh   # 👈 Training starten (nur ausführen)
├── start_eval.sh       # 👈 Modell testen (50 Durchläufe)
├── start_demo.sh       # 👈 Live-Fenster ansehen (3 Episoden)
├── mujoco_arm.xml      # 3D-Welt: Arm, Räder, Würfel, Box
├── arm_env.py          # Die Spielregeln (Gymnasium Environment)
├── train.py            # Das Trainingsskript (PPO)
├── eval.py             # Auswertung des Modells
├── demo.py             # Live-Demo mit Fenster
├── ppo_arm_logs_final/ # 🎯 Das trainierte Modell (hier liegt es)
│   └── ppo_wheel_arm_final_v2   ← das fertige KI-Modell
├── README.md           # Diese Datei
└── .venv/              # Python-Pakete (wird erstellt bei Setup)
```

---

## 🛠️ Setup (einmalig, falls .venv fehlt)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco gymnasium stable-baselines3 torch tensorboard tqdm rich
```

---

## 📊 Aktueller Stand (CEO-Entscheidung)

| Version | Roboter | Aufgabe | Status | Erfolgsrate |
|---------|---------|---------|--------|-------------|
| V1 | 3-Achsen-Arm auf Rädern | Würfel greifen + in Box legen | ✅ Fertig | ~22% |
| V2 | Arm + 3 Werkzeuge (Greifer, Sauger, Schaufel) | 3 Objekte, 3 Boxen | 🔧 Environment fertig, Lernrate noch zu niedrig | 0% (braucht mehr Training) |
| V3 | Roboterhund (Unitree A1) | Laufen lernen | 📋 Geplant (braucht Isaac Lab + RTX GPU) | — |

**Empfehlung:** Fang mit V1 an. Es funktioniert sofort. Wenn du mehr willst, sag Bescheid — ich mache V2 fertig oder starte V3 mit dem Hund.

---

## 🐛 Bekannte Einschränkungen

- 22% Erfolgsrate ist ein guter Start, aber noch nicht perfekt
- Für 100% braucht es mehr Training oder ein feineres Curriculum (Würfel startet immer weiter weg)
- Die 3 Werkzeuge in V2 sind gebaut, aber die KI lernt noch nicht zuverlässig zwischen ihnen zu wechseln

---

## 📄 Lizenz

Privates Lernprojekt. MuJoCo: Apache 2.0. Stable Baselines3: MIT.
