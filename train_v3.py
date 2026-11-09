"""PPO-Training für den Roboterhund V3 (Quadruped Laufenlernen).

Hinweis: Für maximale Performance mit Isaac Lab + RTX GPU siehe Isaac Lab Dokumentation.
Diese Version läuft auf MuJoCo CPU für schnelle Experimente.
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from quad_env import QuadrupedEnv


def main():
    parser = argparse.ArgumentParser(description="PPO-Training für Roboterhund V3")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Anzahl Training-Timesteps")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Gerät für Training")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum-Stufe (0=flach, 1=uneben, 2=hindernisse)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Anzahl paralleler Umgebungen")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    print(f"==============================================")
    print(f"  PPO-Training V3 — Roboterhund (Quadruped)")
    print(f"==============================================")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Gerät:        {device.upper()}")
    print(f"  Curriculum:   Stufe {args.curriculum}")
    print(f"  Parallele Envs: {args.n_envs}")
    print(f"==============================================")
    print()

    log_dir = "./ppo_quadruped_logs"
    os.makedirs(log_dir, exist_ok=True)

    train_env = make_vec_env(
        QuadrupedEnv,
        n_envs=args.n_envs,
        seed=42,
        env_kwargs={"curriculum_stage": args.curriculum},
        wrapper_class=lambda e: Monitor(e),
    )

    eval_env = QuadrupedEnv(render_mode=None, curriculum_stage=args.curriculum)
    eval_env = Monitor(eval_env)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=15000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )

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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device=device,
    )

    print(f"Starte PPO-Training V3 auf {device.upper()}...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "ppo_quadruped_final"))
    print(f"\n✅ Training V3 abgeschlossen!")
    print(f"   Modell gespeichert in: {log_dir}/ppo_quadruped_final")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
