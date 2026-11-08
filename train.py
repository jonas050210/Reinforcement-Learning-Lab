"""PPO-Training für den Wheel-Arm (greifen + in Box legen)."""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from arm_env import WheelArmEnv


def main():
    log_dir = "./ppo_arm_logs_final"
    os.makedirs(log_dir, exist_ok=True)

    # Curriculum Stage 0: Würfel startet direkt vor dem Greifer
    curriculum_stage = 0

    # Vektorisierte Umgebungen (mehrere parallele Instanzen)
    n_envs = 4
    train_env = make_vec_env(
        WheelArmEnv,
        n_envs=n_envs,
        seed=42,
        env_kwargs={"curriculum_stage": curriculum_stage},
        wrapper_class=lambda e: Monitor(e),
    )

    # Eval-Umgebung (einzeln)
    eval_env = WheelArmEnv(render_mode=None, curriculum_stage=curriculum_stage)
    eval_env = Monitor(eval_env)

    # Stop bei guter Belohnung
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=50.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )

    # PPO Agent
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
        tensorboard_log=log_dir,
        device="cpu",
    )

    print("Starte PPO-Training (FINAL, 500k Timesteps)...")
    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
        progress_bar=True,
    )

    # Finales Modell speichern
    model.save(os.path.join(log_dir, "ppo_wheel_arm_final_v2"))
    print("Training abgeschlossen. Modell gespeichert in", log_dir)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
