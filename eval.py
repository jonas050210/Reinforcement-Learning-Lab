"""Evaluierung des trainierten PPO-Modells für den Wheel-Arm V1."""

import argparse
from stable_baselines3 import PPO
from arm_env import WheelArmEnv
import numpy as np


def evaluate(model_path, n_episodes=50, render=False):
    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = WheelArmEnv(render_mode=render_mode)

    successes = 0
    lifts = 0
    touches = 0
    rewards = []
    ep_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_lifted = False
        ep_touched = False
        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if info.get("touching", 0) > 0:
                ep_touched = True
            if info.get("lifted", 0) > 0:
                ep_lifted = True
            if term:
                successes += 1
                break
        if ep_lifted:
            lifts += 1
        if ep_touched:
            touches += 1
        rewards.append(ep_reward)
        ep_lengths.append(step + 1)

    env.close()

    print(f"\n==============================================")
    print(f"  Evaluation: {model_path}")
    print(f"==============================================")
    print(f"  Episoden:           {n_episodes}")
    print(f"  Erfolgsrate (Box):  {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")
    print(f"  Würfel gehoben:     {lifts}/{n_episodes} ({lifts/n_episodes*100:.0f}%)")
    print(f"  Würfel berührt:     {touches}/{n_episodes} ({touches/n_episodes*100:.0f}%)")
    print(f"  Ø Belohnung:        {np.mean(rewards):.1f}")
    print(f"  Beste Belohnung:    {np.max(rewards):.1f}")
    print(f"  Ø Episode-Länge:    {np.mean(ep_lengths):.0f} / {env.max_episode_steps}")
    print(f"==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluiert das trainierte Arm-Modell")
    parser.add_argument("--model", default="./ppo_arm_logs_final/ppo_wheel_arm_final_v2",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=50, help="Anzahl Test-Episoden")
    parser.add_argument("--render", action="store_true", help="Live-Fenster anzeigen")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)
