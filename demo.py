"""Live-Demo: Trainiertes Modell mit MuJoCo-Visualisierung ausführen."""

import time
from stable_baselines3 import PPO
from arm_env import WheelArmEnv


def demo(model_path, n_episodes=5, delay=0.02):
    model = PPO.load(model_path)
    env = WheelArmEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n=== Demo-Episode {ep + 1}/{n_episodes} ===")
        ep_reward = 0
        success = False

        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(delay)

            if info.get("in_box", 0) > 0:
                print(f"  Schritt {step}: Würfel in Box! Erfolg!")
                success = True
                break

            if term:
                print(f"  Schritt {step}: Episode beendet (Erfolg).")
                success = True
                break

        if not success:
            print(f"  Episode {ep + 1}: nicht geschafft (Ø Belohnung: {ep_reward:.1f})")

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live-Demo des trainierten Arm-Modells")
    parser.add_argument("--model", default="./ppo_arm_logs_final/ppo_wheel_arm_best.zip",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Demo-Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    args = parser.parse_args()

    demo(args.model, args.episodes, args.delay)
