#!/usr/bin/env python3
"""
3D-Ansicht V3: Roboterhund im MuJoCo-Viewer ansehen.
Kein trainiertes Modell nötig — öffnet nur die 3D-Welt.
"""

import time
import argparse
from quad_env import QuadrupedEnv


def view(curriculum_stage=0, n_episodes=3, delay=0.02):
    env = QuadrupedEnv(render_mode="human", curriculum_stage=curriculum_stage)

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n=== 3D-Ansicht V3 — Episode {ep + 1}/{n_episodes} (Curriculum Stufe {curriculum_stage}) ===")
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
    parser = argparse.ArgumentParser(description="3D-Ansicht V3 (ohne Modell)")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    view(args.curriculum, args.episodes, args.delay)
