"""Live-Demo: Trainiertes V3-Modell (Roboterhund, 12 DOF) mit MuJoCo-Visualisierung ausführen."""

import time
import argparse
from stable_baselines3 import PPO
from quad_env import QuadrupedEnv


def demo(model_path, n_episodes=3, delay=0.02, curriculum_stage=0):
    model = PPO.load(model_path)
    env = QuadrupedEnv(render_mode="human", curriculum_stage=curriculum_stage)

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n=== Demo-Episode {ep + 1}/{n_episodes} (Curriculum Stufe {curriculum_stage}) ===")
        ep_reward = 0
        distance = 0.0

        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            distance = info.get("distance", 0.0)
            env.render()
            time.sleep(delay)

            if term:
                print(f"  Schritt {step}: Hund umgefallen. Episode beendet.")
                break

        print(f"  Episode {ep + 1}: Distanz {distance:.2f} m, Ø Belohnung: {ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live-Demo des trainierten V3 Roboterhund-Modells")
    parser.add_argument("--model", default="./ppo_quadruped_logs/ppo_quadruped_final",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Demo-Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    args = parser.parse_args()

    demo(args.model, args.episodes, args.delay, args.curriculum)
