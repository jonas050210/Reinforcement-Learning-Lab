#!/usr/bin/env python3
"""
================================================================================
  3D-Ansicht V1 — Einfacher Arm auf Rädern (Blender-Style)
================================================================================
Öffnet die 3D-Welt im MuJoCo-Viewer mit Blender-ähnlicher Kamera.
Kein trainiertes Modell nötig — nur die 3D-Welt ansehen.

Daten werden im Terminal angezeigt (Seiten-Panel Stil).
Steuerung im MuJoCo-Fenster:
  Linke Maustaste + Ziehen  = Kamera rotieren
  Rechte Maustaste + Ziehen = Kamera bewegen
  Scrollrad                 = Zoom ein/aus
================================================================================
"""

import time
import argparse
import numpy as np
from arm_env import WheelArmEnv


def _setup_blender_camera(viewer, env):
    """
    Setzt eine Blender-ähnliche Kamera-Position:
    - Blick von oben-vorn auf die Szene
    - Guter Überblick über Arm, Würfel und Box
    """
    cam = viewer.cam
    # Kamera schaut auf den Mittelpunkt der Szene (Arm-Basis + Würfel + Box)
    cam.lookat[0] = 0.5   # X: zwischen Arm-Basis (0) und Box (0.7)
    cam.lookat[1] = -0.1  # Y: leicht vorne
    cam.lookat[2] = 0.15  # Z: leicht über dem Boden
    cam.distance = 1.2    # Distanz zur Szene
    cam.azimuth = 45      # Azimuth-Winkel (Blender-Style: 45 Grad von vorn)
    cam.elevation = -30   # Elevation: von oben auf die Szene


def view(n_episodes=3, delay=0.02):
    env = WheelArmEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()

        # Kamera beim ersten Mal einrichten (nach erstem render)
        if env._viewer is None:
            env.render()  # Viewer erstellen
        _setup_blender_camera(env._viewer, env)

        print("\n" + "=" * 60)
        print(f"  🖥️  3D-ANSICHT V1 — Episode {ep + 1}/{n_episodes}")
        print("=" * 60)
        print("  Kamera: Blender-Style (45° Azimuth, -30° Elevation)")
        print("  Steuerung: Linke Maustaste = Rotieren | Rechte = Bewegen | Scroll = Zoom")
        print("=" * 60)

        step = 0
        for step in range(env.max_episode_steps):
            action = env.action_space.sample()  # Zufällige Bewegung zum Anschauen
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            time.sleep(delay)

            # Alle 20 Schritte Daten anzeigen (Seiten-Panel Stil)
            if step % 20 == 0 or term:
                d_gc = info.get("dist_gripper_cube", 0)
                d_cb = info.get("dist_cube_box", 0)
                print(f"  Schritt {step:4d} | Würfel-Greifer: {d_gc:.3f} m | Würfel-Box: {d_cb:.3f} m | "
                      f"Belohnung: {reward:6.1f}")

            if term or trunc:
                print(f"  ✅ Episode {ep + 1} beendet nach {step + 1} Schritten")
                break

    env.close()
    print("\n  🖥️  3D-Ansicht V1 beendet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V1 (Blender-Style, ohne Modell)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.episodes, args.delay)
