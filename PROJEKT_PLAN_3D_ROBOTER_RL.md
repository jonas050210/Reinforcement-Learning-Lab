# 🤖 Reinforcement-Learning-Lab: 3D-Roboter-Projekt (Planungsstand)

**Status:** Nur Planung und Internet-Recherche. Noch keine Modelle gebaut, noch kein Code geschrieben.
**Ziel:** Einen 3D-Roboter (Arm, Hund oder Mensch mit Gelenken) in der Simulation mit KI/Reinforcement Learning (RL) trainieren, damit er Ziele selbstständig lernt (z. B. etwas greifen und in eine Box legen, oder laufen lernen).

---

## 1. Die Kernidee

Wir bauen eine simulierte 3D-Umgebung mit einem Roboter, definieren eine Aufgabe (Target/Goal), und trainieren eine KI (meist PPO oder SAC) mit Trial-and-Error, bis der Roboter die Aufgabe löst.

Gewünschte "3 Versionen":
1. **Roboterarm auf Rädern** (mobil), 3 austauschbare Werkzeuge, Aufgabe: Gegenstände aufsammeln und in eine Kiste legen.
2. **Roboterhund (4 Beine)** lernt zuerst laufen, dann weitere Basics (drehen, stehen bleiben, Gelände).
3. **(Optional)** 3D-Mensch mit Gelenken (Humanoid) oder Kombination Arm + mobiler Roboter.

---

## 2. Empfohlene Simulatoren & Engines (Rechercheergebnis)

| Simulator | Lizenz | GPU-Parallelisierung | Stärken | Schwächen | Für unser Projekt |
|---|---|---|---|---|---|
| **MuJoCo + MJX** | Apache 2.0 (kostenlos) | Ja (via MJX auf GPU) | Beste Kontaktphysik für Greifen/Manipulation, riesige Modell-Bibliothek (Menagerie), einfach zu installieren, läuft auch auf CPU | Nicht fotorealistisch, eher Forschung als "Spiel" | ✅ Beste Wahl für Version 1 (Arm greift in Box) |
| **NVIDIA Isaac Lab + Isaac Sim** | BSD-3 / NVIDIA EULA | Ja (4096+ Umgebungen parallel auf einer RTX-GPU) | Beste Durchsatz für Lauf-Roboter (Hund/Humanoid), fotorealistisch, Domain-Randomization, viele fertige Roboter (Unitree A1/Go1/Go2, H1, Franka, UR10e) | Braucht NVIDIA RTX-GPU, Installation aufwendig (~50 GB) | ✅ Beste Wahl für Version 2 (Hund laufen lernen) |
| **PyBullet** | zlib (kostenlos) | Nein (CPU) | Sehr einfach, Python-freundlich, viele Tutorials, gut zum Lernen | Veraltet, schlechtere Physik, wenig Durchsatz | ⚠️ Nur als Einstieg/Lern-Sandbox |
| **Webots** | Open Source | Begrenzt | Einfache GUI, viele fertige Roboter, gut für Bildung und mobile Roboter | Nicht konkurrenzfähig für modernes RL-Training | ❌ Eher nicht |
| **Unity ML-Agents** | Proprietär (kostenlos für kleine Projekte) | Begrenzt | Gut für 3D-Visualisierung, Game-Engine, Plugins | RL-Support schwächer als MuJoCo/Isaac | ❌ Eher nicht |

**Empfehlung:** Starte mit **MuJoCo** (einfach, kostenlos, CPU reicht) für den Arm, und wechsle ggf. auf **Isaac Lab** für den Lauf-Roboterhund, wenn eine NVIDIA-GPU verfügbar ist.

### RL-Bibliotheken & Algorithmen
- **PPO (Proximal Policy Optimization):** Standard-Algorithmus für fortlaufende Robotik-Aufgaben (Laufen, Greifen). In `Stable Baselines3`, `RSL-RL` (Isaac Lab), `skrl`, `rl_games`.
- **SAC / DDPG + HERS (Hindsight Experience Replay):** Gut für Greif-Aufgaben mit Zielkoordinaten (z. B. "Greif den Würfel und lege ihn in die Box").
- **Gymnasium / Gym Robotics:** Standard-Schnittstelle für RL-Umgebungen (Fetch Reach/Push/PickAndPlace/Slide als Vorbild).

---

## 3. 3D-Modelle & Assets (kostenlos / Open Source)

### Roboterarm (für Version 1)
- **Franka Emika Panda / UR5e / UR10e:** Industriestandard, URDF-Modelle verfügbar, in Isaac Lab und Robosuite vorgefertigt.
- **SO-100:** Günstiger Open-Source-Arm, beliebtes RL-Target (z. B. Cube Lifting in Isaac Lab).
- **Open-Source CAD-Arm (5-DOF, URDF + STL, ROS 2):** `srivardhan02/open-robotic-arm` auf GitHub (URDF, STL-Meshes, Parallel-Greifer, RViz).
- **6-DOF Arduino-Arm:** STLs auf Cults3D/Thingiverse (omartronics, 5 Rotationsachsen + Greifer, MG996R/SG90 Servos).
- **Greifer/Werkzeuge:** 3-Finger-Greifer (Robotiq), Adaptive Gripper, Magnet, Sauger, Schaufel — als separate STL/STEP-Modelle auf GrabCAD, Cults3D, Printables, Thingiverse.

### Roboterhund (für Version 2)
- **Unitree A1:** 12-DOF (3 Gelenke pro Bein), offenes URDF, Klassiker für akademische Locomotion-Papiere (Berkeley RMA, "Learning to Walk in 20 Minutes").
- **Unitree Go1 / Go2:** Nachfolger des A1, ebenfalls URDF verfügbar.
- **ANYmal (ANYbotics):** Etwas größer, in Isaac Lab vorintegriert.

### 3D-Mensch / Humanoid (für Version 3)
- **Unitree H1 / G1:** Offen, in Isaac Lab unterstützt, ganze-body RL-Tracks verfügbar.
- **MuJoCo Menagerie:** Kostenlose Sammlung an MJCF-Modellen (Arme, Hände, Humanoids, Tiere).
- **Boston Dynamics (Atlas/Spot):** Nicht frei verfügbar, nur als Referenz.

### Quellen für 3D-Modelle
- **URDF/MJCF:** GitHub, MuJoCo Menagerie, Isaac Lab/Assets, ROS 2-Repositories.
- **STL/STEP/OBJ (3D-Druck/CAD):** GrabCAD, Thingiverse, Printables, Cults3D, CGTrader, MyMiniFactory.
- **Formate:** URDF (Roboter-Beschreibung, XML), MJCF (MuJoCo), USD (Isaac Sim/Omniverse), STL (3D-Druck/Mesh).

### Blender / 3D-Tools & Plugins
- **Blender:** Kostenlos, kann STL/OBJ importieren und als Collada/GLTF exportieren, das viele Simulatoren lesen können.
- **URDF-Exporter für Blender:** Blender-Add-ons, um URDF aus Blender-Modellen zu erzeugen.
- **Meshroom / Agisoft:** Falls reale Objekte eingescannt werden sollen.

---

## 4. Die 3 Versionen im Detail

### Version 1: Roboterarm auf Rädern + 3 Werkzeuge (Manipulation)
**Roboter:** 3-Achsen-Arm (Base, Schulter, Ellbogen) auf einer fahrbaren Basis (Räder), damit er sich langsam zum Objekt bewegen kann.
**Werkzeuge (3 Stück, wechselbar):**
1. Parallel-Greifer (klassisch greifen)
2. Sauger/Magnet (für glatte/metallische Objekte)
3. Schaufel/Schaufelgreifer (für lose Teile/Kleinkram)

**Aufgabe:** Roboter fährt zu einem Gegenstand, wählt/aktiviert das richtige Werkzeug, hebt den Gegenstand auf und legt ihn in eine Box/Zielzone.
**Belohnung (Reward):** Annähern an Objekt (+), Greifen (+), Heben (+), Ablegen in Box (+++), Fallenlassen (-).
**Simulator:** MuJoCo (CPU reicht) oder Isaac Lab.
**RL-Algo:** PPO oder DDPG/SAC + HER.

### Version 2: Roboterhund lernt laufen (Locomotion)
**Roboter:** Unitree A1 oder Go1 (12 Gelenke, 4 Beine).
**Aufgaben-Stufen:**
1. **Stehen:** Roboter soll nicht umfallen.
2. **Vorwärts laufen:** Zielgeschwindigkeit einhalten (Velocity Tracking).
3. **Drehen & Kurven:** Yaw-Befehle ausführen.
4. **Unebenes Terrain:** Auf Schaumstoff, Matte, kleine Hindernisse laufen (Domain Randomization).
**Belohnung:** Vorwärtsgeschwindigkeit (+), gerade Haltung/kein Umfallen (+), geringer Energieverbrauch (+), Umfallen (---).
**Simulator:** Isaac Lab (beste Wahl für Locomotion, GPU nötig) oder MuJoCo.
**RL-Algo:** PPO (RSL-RL oder Stable Baselines3).

### Version 3: 3D-Mensch mit Gelenken (Humanoid) — optional/Später
**Roboter:** Unitree H1 oder G1 Humanoid.
**Aufgaben:** Aufrechtes Stehen, Gehen, einfache Greif-Bewegung mit den Armen.
**Hinweis:** Deutlich schwieriger als Arm oder Hund (Balance, viele Gelenke). Erst nach Erfolg mit V1/V2 angehen.
**Simulator:** Isaac Lab empfohlen.

---

## 5. Vorgeschlagene Roadmap (Meilensteine)

| Phase | Woche | Meilenstein |
|---|---|---|
| 1 | 1–2 | Simulator installieren (MuJoCo oder Isaac Lab), Testumgebung mit leerem Roboter laufen lassen |
| 2 | 3–4 | 3D-Modell importieren (URDF/MJCF), Gelenke und Grenzen prüfen, einfache Szene (Boden, Box, Gegenstand) bauen |
| 3 | 5–7 | RL-Umgebung bauen (Observations, Actions, Rewards, Resets), erstes PPO-Training |
| 4 | 8–10 | Version 1 fertig: Arm greift Objekt und legt in Box (Erfolgsrate messen) |
| 5 | 11–13 | Version 2: Roboterhund läuft stabil auf flachem Boden, dann leichtes Gelände |
| 6 | 14+ | Version 3 (optional): Humanoid oder Kombination Arm + mobiler Roboter |

---

## 6. Hardware-Anforderungen

| Komponente | Minimum | Empfohlen |
|---|---|---|
| CPU | 4 Kerne | 8+ Kerne |
| RAM | 8 GB | 16–32 GB |
| GPU | Nicht nötig (MuJoCo auf CPU) | NVIDIA RTX 3070+ (8 GB VRAM) für Isaac Lab |
| Speicher | 20 GB | 100 GB (Isaac Sim/Isaac Lab ist groß) |
| OS | Linux (empfohlen), Windows geht mit WSL2 | Linux (Ubuntu) für Isaac Lab |

---

## 7. Nächste Schritte — Entscheidung nötig

Damit ich weitermachen kann, bräuchte ich kurz eine Rückmeldung:

1. **Welche Version wollen wir zuerst umsetzen?** (V1 Arm mit Werkzeugen, V2 Hund laufen lernen, oder beides parallel planen?)
2. **Welche Hardware steht zur Verfügung?** (Besonders: NVIDIA-GPU vorhanden? Damit entscheiden wir MuJoCo vs. Isaac Lab.)
3. **Soll das Projekt später auf einen echten Roboter übertragbar sein (Sim-to-Real), oder bleibt es rein in der Simulation?**

Sobald das geklärt ist, kann ich:
- Den Simulator installieren und ein erstes Test-Szenario anlegen,
- Die passenden 3D-Modelle (URDF/STL) heraussuchen und in den Repo laden,
- Die erste RL-Umgebung (Code) aufsetzen.

Alle Quellen und Links zur Recherche stehen oben im Dokument. Möchtest du, dass ich eine der Versionen als ersten konkreten Schritt ausarbeite?
