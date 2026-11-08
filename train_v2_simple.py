"""V2 Vereinfacht: Nur Greifer, Objekt startet direkt vor dem Arm."""

import os
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from arm_env_v2 import WheelArm3ToolsEnv


def main():
    log_dir = "./ppo_arm_v2_simple_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Klasse mit festem Greifer und Objekt direkt vor dem Arm
    class SimpleEnv(WheelArm3ToolsEnv):
        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed)
            # Greifer fest vorgewählt (kein Wechsel nötig)
            self.current_tool = self.TOOL_GRIPPER
            # Objekt direkt vor Greifer platzieren
            tool_pos = self._get_tool_pos()
            cube_addr = self._object_qpos_addr("cube1")
            self.data.qpos[cube_addr : cube_addr + 7] = np.array([
                tool_pos[0], tool_pos[1], tool_pos[2] + 0.02, 1, 0, 0, 0
            ])
            mujoco.mj_forward(self.model, self.data)
            return self._get_obs(), info

    n_envs = 4
    train_env = make_vec_env(
        SimpleEnv,
        n_envs=n_envs,
        seed=42,
        wrapper_class=lambda e: Monitor(e),
    )
    eval_env = SimpleEnv(render_mode=None)
    eval_env = Monitor(eval_env)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=80.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
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
        tensorboard_log=log_dir,
        device="cpu",
    )

    print("Starte V2-Einfach-Training (nur Greifer, Objekt direkt davor)...")
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "ppo_arm_v2_simple_final"))
    print("Fertig. Modell in", log_dir)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
