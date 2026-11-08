"""PPO-Training für den Wheel-Arm V1-Optimiert (Ubuntu + GPU Ready).

Trainiert einen 3-Achsen-Roboterarm auf fahrbarer Basis, damit er einen Würfel
greift und in eine Zielbox legt.

Verwendung:
    python train.py                    # CPU-Modus
    python train.py --device cuda      # GPU-Modus (RTX 4060 Ti)
    python train.py --curriculum 2     # Mit Curriculum-Stufe 2
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from arm_env import WheelArmEnv


def make_env(render_mode=None, rank=0, curriculum_stage=0):
    def _init():
        env = WheelArmEnv(
            render_mode=render_mode,
            curriculum_stage=curriculum_stage,
        )
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO-Training für Wheel-Arm V1")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Anzahl Training-Timesteps (default: 500.000)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Gerät für Training (default: auto)")
    parser.add_argument("--curriculum", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Curriculum-Stufe (0=einfach, 3=schwer)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Anzahl paralleler Umgebungen")
    args = parser.parse_args()

    # Gerät bestimmen
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    print(f"==============================================")
    print(f"  PPO-Training V1 — WheelArmEnv")
    print(f"==============================================")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Gerät:        {device.upper()}")
    print(f"  Curriculum:   Stufe {args.curriculum}")
    print(f"  Parallele Envs: {args.n_envs}")
    print(f"==============================================")
    print()

    log_dir = "./ppo_arm_logs_final"
    os.makedirs(log_dir, exist_ok=True)

    # Vektorisierte Umgebungen
    train_env = make_vec_env(
        WheelArmEnv,
        n_envs=args.n_envs,
        seed=42,
        env_kwargs={"curriculum_stage": args.curriculum},
        wrapper_class=lambda e: Monitor(e),
    )

    # Eval-Umgebung (einzeln)
    eval_env = WheelArmEnv(
        render_mode=None,
        curriculum_stage=args.curriculum,
    )
    eval_env = Monitor(eval_env)

    # Callback: Stop bei guter Belohnung
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=15000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )

    # PPO Agent mit optimierten Parametern
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # Etwas mehr Exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device=device,
    )

    print(f"Starte PPO-Training auf {device.upper()}...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # Finales Modell speichern
    model.save(os.path.join(log_dir, "ppo_wheel_arm_final_v2"))
    print(f"\n✅ Training abgeschlossen!")
    print(f"   Modell gespeichert in: {log_dir}/ppo_wheel_arm_final_v2")
    print(f"\n   Jetzt testen:")
    print(f"     ./start_eval.sh")
    print(f"     ./start_demo.sh")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
