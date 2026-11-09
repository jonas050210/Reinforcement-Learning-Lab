"""Evaluierung des trainierten PPO-Modells für den Roboterhund V3 (Quadruped, 12 DOF)."""

import argparse
from stable_baselines3 import PPO
from quad_env import QuadrupedEnv
import numpy as np


def evaluate(model_path, n_episodes=50, render=False, curriculum_stage=0):
    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = QuadrupedEnv(render_mode=render_mode, curriculum_stage=curriculum_stage)

    distances = []
    rewards = []
    ep_lengths = []
    fallen_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term:
                fallen_count += 1
                break

        distances.append(info.get("distance", 0.0))
        rewards.append(ep_reward)
        ep_lengths.append(step + 1)

    env.close()

    print(f"\n==============================================")
    print(f"  Evaluation V3 — {model_path}")
    print(f"==============================================")
    print(f"  Episoden:           {n_episodes}")
    print(f"  Curriculum-Stufe:   {curriculum_stage}")
    print(f"  Umgefallen:         {fallen_count}/{n_episodes} ({fallen_count/n_episodes*100:.0f}%)")
    print(f"  Ø Distanz:          {np.mean(distances):.2f} m")
    print(f"  Max. Distanz:       {np.max(distances):.2f} m")
    print(f"  Ø Belohnung:        {np.mean(rewards):.1f}")
    print(f"  Beste Belohnung:    {np.max(rewards):.1f}")
    print(f"  Ø Episode-Länge:    {np.mean(ep_lengths):.0f} / {env.max_episode_steps}")
    print(f"==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluiert das trainierte V3 Roboterhund-Modell")
    parser.add_argument("--model", default="./ppo_quadruped_logs/ppo_quadruped_final",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=50, help="Anzahl Test-Episoden")
    parser.add_argument("--render", action="store_true", help="Live-Fenster anzeigen")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render, args.curriculum)
