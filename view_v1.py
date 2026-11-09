#!/usr/bin/env python3
"""
3D-Ansicht V1: Einfacher Arm auf Rädern im MuJoCo-Viewer ansehen.
Kein trainiertes Modell nötig — öffnet nur die 3D-Welt.
"""

import time
import argparse
from arm_env import WheelArmEnv


def view(n_episodes=3, delay=0.02):
    env = WheelArmEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n=== 3D-Ansicht V1 — Episode {ep + 1}/{n_episodes} ===")
        print("Schließe das MuJoCo-Fenster oder warte, um fortzufahren.")

        for step in range(env.max_episode_steps):
            action = env.action_space.sample()  # Zufällige Bewegung zum Anschauen
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            time.sleep(delay)

            if term or trunc:
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-Ansicht V1 (ohne Modell)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.episodes, args.delay)
