#!/usr/bin/env python3
"""
================================================================================
  3D-Ansicht V2 — Arm mit 3 Werkzeugen (Blender-Style)
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
from arm_env_v2 import WheelArm3ToolsEnv


def _setup_blender_camera(viewer, env):
    """
    Setzt eine Blender-ähnliche Kamera-Position für V2:
    - Überblick über Arm, 3 Objekte, 3 Boxen und Hindernisse
    """
    cam = viewer.cam
    cam.lookat[0] = 0.8   # X: Mitte der Szene (Objekte und Boxen)
    cam.lookat[1] = 0.1   # Y: leicht vorne
    cam.lookat[2] = 0.2   # Z: über dem Boden
    cam.distance = 1.8    # Etwas weiter weg wegen mehr Objekten
    cam.azimuth = 45      # Blender-Style
    cam.elevation = -25   # Von oben auf die Szene


def view(curriculum_stage=0, n_episodes=3, delay=0.02):
    env = WheelArm3ToolsEnv(render_mode="human", curriculum_stage=curriculum_stage)

    for ep in range(n_episodes):
        obs, info = env.reset()

        if env._viewer is None:
            env.render()
        _setup_blender_camera(env._viewer, env)

        tool_names = {0: "Greifer", 1: "Sauger", 2: "Schaufel"}

        print("\n" + "=" * 60)
        print(f"  🖥️  3D-ANSICHT V2 — Episode {ep + 1}/{n_episodes}")
        print(f"  Curriculum-Stufe: {curriculum_stage} | Aktive Objekte: {env.n_active_objects}")
        print("=" * 60)
        print("  Kamera: Blender-Style (45° Azimuth, -25° Elevation)")
        print("  Steuerung: Linke Maustaste = Rotieren | Rechte = Bewegen | Scroll = Zoom")
        print("=" * 60)

        step = 0
        for step in range(env.max_episode_steps):
            action = env.action_space.sample()  # Zufällige Bewegung
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            time.sleep(delay)

            # Daten alle 20 Schritte (Seiten-Panel Stil)
            if step % 20 == 0 or term:
                tool_name = tool_names.get(env.current_tool, "?")
                d_obj = info.get("dist_to_obj", 0)
                d_box = info.get("dist_to_box", 0)
                n_in_box = info.get("objects_in_box", 0)
                print(f"  Schritt {step:4d} | Werkzeug: {tool_name:8s} | "
                      f"Objekt: {d_obj:.3f} m | Box: {d_box:.3f} m | "
                      f"In Boxen: {n_in_box}/{env.n_active_objects} | Belohnung: {reward:6.1f}")

            if term or trunc:
                print(f"  ✅ Episode {ep + 1} beendet nach {step + 1} Schritten "
                      f"({n_in_box}/{env.n_active_objects} Objekte in Boxen)")
                break

    env.close()
    print("\n  🖥️  3D-Ansicht V2 beendet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V2 (Blender-Style, ohne Modell)")
    parser.add_argument("--curriculum", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Curriculum-Stufe (0=einfach, 3=schwer)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.curriculum, args.episodes, args.delay)
