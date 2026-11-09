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
  ESC                       = Fenster schließen
================================================================================
"""

import time
import argparse
import sys
import numpy as np
from quad_env import QuadrupedEnv

GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _setup_blender_camera(viewer):
    """
    Setzt eine Blender-ähnliche Kamera-Position für V3:
    - Seitenansicht auf den Roboterhund (klassische Locomotion-Ansicht)
    """
    cam = viewer.cam
    cam.lookat[0] = 0.0
    cam.lookat[1] = 0.5
    cam.lookat[2] = 0.35
    cam.distance = 1.5
    cam.azimuth = 20
    cam.elevation = -10


def view(curriculum_stage=0, n_episodes=3, delay=0.02):
    env = QuadrupedEnv(render_mode="human", curriculum_stage=curriculum_stage)

    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("  🖥️  3D-ANSICHT V3 — Roboterhund (Quadruped) (Blender-Style)")
    print("=" * 70)
    print(f"{RESET}")
    print(f"{BLUE}  Kamera:{RESET} Blender-Style (20° Azimuth, -10° Elevation)")
    print(f"{BLUE}  Steuerung:{RESET} Linke Maustaste = Rotieren | Rechte = Bewegen | Scroll = Zoom")
    print(f"{BLUE}  Curriculum-Stufe:{RESET} {curriculum_stage}")
    print(f"{BLUE}  Episoden:{RESET} {n_episodes} | Verzögerung: {delay*1000:.0f} ms")
    print()

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()

            if env._viewer is None:
                env.render()
            _setup_blender_camera(env._viewer)

            print(f"{YELLOW}{BOLD}  ▶ Episode {ep + 1}/{n_episodes}{RESET}")
            print(f"  {'─' * 68}")

            step = 0
            ep_reward = 0.0
            max_vx = 0.0
            for step in range(env.max_episode_steps):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                env.render()
                time.sleep(delay)

                ep_reward += reward

                vx = info.get("vx", 0)
                if abs(vx) > abs(max_vx):
                    max_vx = vx

                if step % 20 == 0 or term or trunc:
                    upright = info.get("upright", 0)
                    distance = info.get("distance", 0)
                    fallen = info.get("fallen", False)

                    if fallen:
                        status_color = RED
                        status_text = "UMGEFALLEN"
                    elif upright > 0.8:
                        status_color = GREEN
                        status_text = "STABIL"
                    else:
                        status_color = YELLOW
                        status_text = "LANGSAME"

                    print(f"  Schritt {step:4d} | vx: {vx:6.3f} m/s | Haltung: {upright:5.2f} | "
                          f"Distanz: {distance:6.2f} m | {status_color}{status_text}{RESET} | "
                          f"Belohnung: {reward:6.1f}")

                if term or trunc:
                    print(f"  {GREEN}✅ Episode {ep + 1} beendet nach {step + 1} Schritten{RESET}")
                    print(f"      Maximale vx: {max_vx:.3f} m/s | Distanz: {distance:.2f} m | "
                          f"Gesamte Belohnung: {ep_reward:.1f}")
                    break

            print()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Abbruch durch Benutzer{RESET}")
    finally:
        env.close()
        print(f"{CYAN}{BOLD}  🖥️  3D-Ansicht V3 beendet.{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V3 (Blender-Style, ohne Modell)")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.curriculum, args.episodes, args.delay)
