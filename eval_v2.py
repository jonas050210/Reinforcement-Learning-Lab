"""Evaluierung des trainierten PPO-Modells für den Wheel-Arm V2 (3 Werkzeuge, 3 Objekte)."""

import argparse
from stable_baselines3 import PPO
from arm_env_v2 import WheelArm3ToolsEnv
import numpy as np


def evaluate(model_path, n_episodes=50, render=False, curriculum_stage=0):
    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = WheelArm3ToolsEnv(render_mode=render_mode, curriculum_stage=curriculum_stage)

    successes = 0
    total_objects_in_box = 0
    total_objects_active = 0
    rewards = []
    ep_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term:
                successes += 1
                break

        n_in_box = info.get("objects_in_box", 0)
        n_active = info.get("n_active", 1)
        total_objects_in_box += n_in_box
        total_objects_active += n_active
        rewards.append(ep_reward)
        ep_lengths.append(step + 1)

    env.close()

    print(f"\n==============================================")
    print(f"  Evaluation V2 — {model_path}")
    print(f"==============================================")
    print(f"  Episoden:           {n_episodes}")
    print(f"  Curriculum-Stufe:   {curriculum_stage}")
    print(f"  Komplette Erfolge:  {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    print(f"  Objekte in Boxen:   {total_objects_in_box}/{total_objects_active} "
          f"({total_objects_in_box/max(total_objects_active,1)*100:.0f}%)")
    print(f"  Ø Belohnung:        {np.mean(rewards):.1f}")
    print(f"  Beste Belohnung:    {np.max(rewards):.1f}")
    print(f"  Ø Episode-Länge:    {np.mean(ep_lengths):.0f} / {env.max_episode_steps}")
    print(f"==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluiert das trainierte V2 Arm-Modell")
    parser.add_argument("--model", default="./ppo_arm_v2_logs/ppo_arm_v2_final",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=50, help="Anzahl Test-Episoden")
    parser.add_argument("--render", action="store_true", help="Live-Fenster anzeigen")
    parser.add_argument("--curriculum", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Curriculum-Stufe (0=einfach, 3=schwer)")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render, args.curriculum)
