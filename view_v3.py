#!/usr/bin/env python3
"""
================================================================================
  3D-Ansicht V3 — Roboterhund (Quadruped) (Blender-Style)
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
from quad_env import QuadrupedEnv


def _setup_blender_camera(viewer, env):
    """
    Setzt eine Blender-ähnliche Kamera-Position für V3:
    - Seitenansicht auf den Roboterhund (klassische Locomotion-Ansicht)
    """
    cam = viewer.cam
    cam.lookat[0] = 0.0   # X: Mitte des Hundes
    cam.lookat[1] = 0.5   # Y: seitlich versetzt für gute Bein-Sicht
    cam.lookat[2] = 0.35  # Z: auf Rumpfhöhe
    cam.distance = 1.5    # Distanz für kompletten Hund im Bild
    cam.azimuth = 20      # Fast von der Seite (wie in Blender Locomotion-View)
    cam.elevation = -10   # Leicht von oben


def view(curriculum_stage=0, n_episodes=3, delay=0.02):
    env = QuadrupedEnv(render_mode="human", curriculum_stage=curriculum_stage)

    for ep in range(n_episodes):
        obs, info = env.reset()

        if env._viewer is None:
            env.render()
        _setup_blender_camera(env._viewer, env)

        print("\n" + "=" * 60)
        print(f"  🖥️  3D-ANSICHT V3 — Roboterhund Episode {ep + 1}/{n_episodes}")
        print(f"  Curriculum-Stufe: {curriculum_stage}")
        print("=" * 60)
        print("  Kamera: Blender-Style (20° Azimuth, -10° Elevation)")
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
                vx = info.get("vx", 0)
                upright = info.get("upright", 0)
                distance = info.get("distance", 0)
                fallen = info.get("fallen", False)
                status = "⚠️  UMGEFALLEN" if fallen else "✅ stabil"
                print(f"  Schritt {step:4d} | vx: {vx:6.3f} m/s | Haltung: {upright:5.2f} | "
                      f"Distanz: {distance:6.2f} m | {status} | Belohnung: {reward:6.1f}")

            if term or trunc:
                print(f"  ✅ Episode {ep + 1} beendet nach {step + 1} Schritten "
                      f"(Distanz: {distance:.2f} m)")
                break

    env.close()
    print("\n  🖥️  3D-Ansicht V3 beendet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V3 (Blender-Style, ohne Modell)")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.curriculum, args.episodes, args.delay)
