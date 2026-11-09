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
  ESC                       = Fenster schließen
================================================================================
"""

import time
import argparse
import sys
import numpy as np
from arm_env_v2 import WheelArm3ToolsEnv

GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

TOOL_NAMES = {0: "GREIFER", 1: "SAUGER", 2: "SCHAUFEL"}


def _setup_blender_camera(viewer):
    """
    Setzt eine Blender-ähnliche Kamera-Position für V2:
    - Überblick über Arm, 3 Objekte, 3 Boxen und Hindernisse
    """
    cam = viewer.cam
    cam.lookat[0] = 0.8
    cam.lookat[1] = 0.1
    cam.lookat[2] = 0.2
    cam.distance = 1.8
    cam.azimuth = 45
    cam.elevation = -25


def view(curriculum_stage=0, n_episodes=3, delay=0.02):
    env = WheelArm3ToolsEnv(render_mode="human", curriculum_stage=curriculum_stage)

    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("  🖥️  3D-ANSICHT V2 — Arm mit 3 Werkzeugen (Blender-Style)")
    print("=" * 70)
    print(f"{RESET}")
    print(f"{BLUE}  Kamera:{RESET} Blender-Style (45° Azimuth, -25° Elevation)")
    print(f"{BLUE}  Steuerung:{RESET} Linke Maustaste = Rotieren | Rechte = Bewegen | Scroll = Zoom")
    print(f"{BLUE}  Curriculum-Stufe:{RESET} {curriculum_stage} | Aktive Objekte: {env.n_active_objects}")
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
            for step in range(env.max_episode_steps):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                env.render()
                time.sleep(delay)

                ep_reward += reward

                if step % 20 == 0 or term or trunc:
                    tool_name = TOOL_NAMES.get(env.current_tool, "?")
                    d_obj = info.get("dist_to_obj", 0)
                    d_box = info.get("dist_to_box", 0)
                    n_in_box = info.get("objects_in_box", 0)
                    n_active = info.get("n_active", 1)
                    grasped = info.get("grasped", 0)

                    status_parts = []
                    if grasped:
                        status_parts.append(f"{GREEN}GEHALTEN{RESET}")
                    if n_in_box >= n_active:
                        status_parts.append(f"{GREEN}ALLE IN BOX{RESET}")
                    status_str = " | ".join(status_parts) if status_parts else "—"

                    print(f"  Schritt {step:4d} | Werkzeug: {tool_name:10s} | "
                          f"Objekt: {d_obj:.3f} m | Box: {d_box:.3f} m | "
                          f"In Boxen: {n_in_box}/{n_active} | {status_str}")

                if term or trunc:
                    print(f"  {GREEN}✅ Episode {ep + 1} beendet nach {step + 1} Schritten{RESET}")
                    print(f"      Objekte in Boxen: {n_in_box}/{n_active} | Gesamte Belohnung: {ep_reward:.1f}")
                    break

            print()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Abbruch durch Benutzer{RESET}")
    finally:
        env.close()
        print(f"{CYAN}{BOLD}  🖥️  3D-Ansicht V2 beendet.{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V2 (Blender-Style, ohne Modell)")
    parser.add_argument("--curriculum", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Curriculum-Stufe (0=einfach, 3=schwer)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.curriculum, args.episodes, args.delay)
