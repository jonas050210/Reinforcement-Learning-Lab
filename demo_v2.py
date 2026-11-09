"""Live-Demo: Trainiertes V2-Modell (3 Werkzeuge, 3 Objekte) mit MuJoCo-Visualisierung ausführen."""

import time
import argparse
from stable_baselines3 import PPO
from arm_env_v2 import WheelArm3ToolsEnv


def demo(model_path, n_episodes=3, delay=0.02, curriculum_stage=0):
    model = PPO.load(model_path)
    env = WheelArm3ToolsEnv(render_mode="human", curriculum_stage=curriculum_stage)

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"\n=== Demo-Episode {ep + 1}/{n_episodes} (Curriculum Stufe {curriculum_stage}) ===")
        ep_reward = 0
        success = False

        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(delay)

            if info.get("objects_in_box", 0) >= info.get("n_active", 1):
                print(f"  Schritt {step}: Alle Objekte in Boxen! Erfolg!")
                success = True
                break

            if term:
                print(f"  Schritt {step}: Episode beendet (Erfolg).")
                success = True
                break

        if not success:
            n_in_box = info.get("objects_in_box", 0)
            n_active = info.get("n_active", 1)
            print(f"  Episode {ep + 1}: nicht geschafft ({n_in_box}/{n_active} in Box, Ø Belohnung: {ep_reward:.1f})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live-Demo des trainierten V2 Arm-Modells")
    parser.add_argument("--model", default="./ppo_arm_v2_logs/ppo_arm_v2_final",
                        help="Pfad zum .zip Modell")
    parser.add_argument("--episodes", type=int, default=3, help="Anzahl Demo-Episoden")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Verzögerung pro Schritt (Sekunden)")
    parser.add_argument("--curriculum", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Curriculum-Stufe (0=einfach, 3=schwer)")
    args = parser.parse_args()

    demo(args.model, args.episodes, args.delay, args.curriculum)
