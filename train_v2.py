"""PPO-Training für den Wheel-Arm V2 mit 3 Werkzeugen und mehreren Objekten."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from arm_env_v2 import WheelArm3ToolsEnv


def main():
    log_dir = "./ppo_arm_v2_logs"
    os.makedirs(log_dir, exist_ok=True)

    curriculum_stage = 0  # Start: 1 Objekt, einfach

    # Vektorisierte Umgebungen
    n_envs = 4
    train_env = make_vec_env(
        WheelArm3ToolsEnv,
        n_envs=n_envs,
        seed=42,
        env_kwargs={"curriculum_stage": curriculum_stage},
        wrapper_class=lambda e: Monitor(e),
    )

    # Eval-Umgebung
    eval_env = WheelArm3ToolsEnv(render_mode=None, curriculum_stage=curriculum_stage)
    eval_env = Monitor(eval_env)

    # Stop bei guter Belohnung
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

    print(f"Starte PPO-Training V2 (3 Werkzeuge, {curriculum_stage + 1} Objekt, Curriculum Stage {curriculum_stage})...")
    model.learn(
        total_timesteps=300_000,
        callback=eval_callback,
        progress_bar=True,
    )

    # Finales Modell speichern
    model.save(os.path.join(log_dir, "ppo_arm_v2_final"))
    print("Training abgeschlossen. Modell gespeichert in", log_dir)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
