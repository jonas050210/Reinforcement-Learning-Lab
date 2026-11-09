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
  ESC                       = Fenster schließen

Tastenkürzel im Terminal:
  Strg+C                    = Abbruch
================================================================================
"""

import time
import argparse
import sys
import numpy as np
from arm_env import WheelArmEnv

# ANSI Farben für farbigen Terminal-Output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _setup_blender_camera(viewer):
    """
    Setzt eine Blender-ähnliche Kamera-Position für V1:
    - Blick von oben-vorn auf die Szene
    - Guter Überblick über Arm, Würfel und Box
    """
    cam = viewer.cam
    cam.lookat[0] = 0.5
    cam.lookat[1] = -0.1
    cam.lookat[2] = 0.15
    cam.distance = 1.2
    cam.azimuth = 45
    cam.elevation = -30


def view(n_episodes=3, delay=0.02):
    env = WheelArmEnv(render_mode="human")

    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("  🖥️  3D-ANSICHT V1 — Einfacher Arm auf Rädern (Blender-Style)")
    print("=" * 70)
    print(f"{RESET}")
    print(f"{BLUE}  Kamera:{RESET} Blender-Style (45° Azimuth, -30° Elevation)")
    print(f"{BLUE}  Steuerung:{RESET} Linke Maustaste = Rotieren | Rechte = Bewegen | Scroll = Zoom")
    print(f"{BLUE}  Episoden:{RESET} {n_episodes} | Verzögerung: {delay*1000:.0f} ms")
    print()

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()

            # Viewer erstellen und Kamera einrichten
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

                # Daten alle 20 Schritte oder am Ende
                if step % 20 == 0 or term or trunc:
                    d_gc = info.get("dist_gripper_cube", 0)
                    d_cb = info.get("dist_cube_box", 0)
                    touching = info.get("touching", 0)
                    lifted = info.get("lifted", 0)
                    in_box = info.get("in_box", 0)

                    status_parts = []
                    if touching:
                        status_parts.append(f"{GREEN}BERÜHRUNG{RESET}")
                    if lifted:
                        status_parts.append(f"{MAGENTA}GEHOBEN{RESET}")
                    if in_box:
                        status_parts.append(f"{GREEN}IN BOX{RESET}")
                    status_str = " | ".join(status_parts) if status_parts else "—"

                    print(f"  Schritt {step:4d} | Würfel-Greifer: {d_gc:.3f} m | "
                          f"Würfel-Box: {d_cb:.3f} m | Belohnung: {reward:6.1f} | {status_str}")

                if term or trunc:
                    print(f"  {GREEN}✅ Episode {ep + 1} beendet nach {step + 1} Schritten{RESET}")
                    print(f"      Gesamte Belohnung: {ep_reward:.1f}")
                    break

            print()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️  Abbruch durch Benutzer{RESET}")
    finally:
        env.close()
        print(f"{CYAN}{BOLD}  🖥️  3D-Ansicht V1 beendet.{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V1 (Blender-Style, ohne Modell)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.episodes, args.delay)
