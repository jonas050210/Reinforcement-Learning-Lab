"""
Gymnasium Environment V3: Roboterhund (Quadruped) lernt laufen.

12 DOF (3 pro Bein), Aufgabe: Vorwärts laufen ohne umzufallen.
Belohnung: Vorwärtsgeschwindigkeit, aufrechte Haltung, geringer Energieverbrauch.

Curriculum: Stufen für unterschiedliches Gelände (später erweiterbar).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer as mj_viewer


class QuadrupedEnv(gym.Env):
    """Roboterhund mit 12 Gelenken, lernt vorwärts laufen."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="mujoco_quadruped.xml", render_mode=None,
                 max_episode_steps=1000, curriculum_stage=0, device="cpu"):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage = curriculum_stage
        self.device = device

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # Alle 12 Gelenk-Namen
        self.joint_names = [
            "hip_fl", "thigh_fl", "calf_fl",
            "hip_fr", "thigh_fr", "calf_fr",
            "hip_hl", "thigh_hl", "calf_hl",
            "hip_hr", "thigh_hr", "calf_hr",
        ]
        self.joint_ranges = {
            "hip_fl": (-0.8, 0.8), "thigh_fl": (-1.5, 0.5), "calf_fl": (-0.5, 2.5),
            "hip_fr": (-0.8, 0.8), "thigh_fr": (-1.5, 0.5), "calf_fr": (-0.5, 2.5),
            "hip_hl": (-0.8, 0.8), "thigh_hl": (-1.5, 0.5), "calf_hl": (-0.5, 2.5),
            "hip_hr": (-0.8, 0.8), "thigh_hr": (-1.5, 0.5), "calf_hr": (-0.5, 2.5),
        }
        self.action_scale = np.array([0.08] * 12, dtype=np.float32)

        self._ids = self._get_ids()

        # Action Space: 12 Gelenk-Zielpositionen
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 12, dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space:
        #   trunk_pos (3) + trunk_quat (4) + trunk_vel (6) +
        #   joint_pos (12) + joint_vel (12) +
        #   feet_contact (4) +
        #   target_velocity (2) +
        #   previous_action (12)
        obs_dim = 3 + 4 + 6 + 12 + 12 + 4 + 2 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.prev_trunk_x = 0.0
        self.episode_distance = 0.0

    def _get_ids(self):
        ids = {}
        for name in self.joint_names:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        ids["trunk"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        for foot in ["foot_fl", "foot_fr", "foot_hl", "foot_hr"]:
            ids[foot] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, foot
            )
        return ids

    def _joint_qpos_addr(self, name):
        return int(self.model.jnt_qposadr[self._ids[name]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.prev_trunk_x = 0.0
        self.episode_distance = 0.0

        # Aufsteh-Position: Alle Gelenke auf Standardwerte
        default_positions = {
            "hip_fl": 0.0, "thigh_fl": 0.7, "calf_fl": -1.4,
            "hip_fr": 0.0, "thigh_fr": 0.7, "calf_fr": -1.4,
            "hip_hl": 0.0, "thigh_hl": 0.7, "calf_hl": -1.4,
            "hip_hr": 0.0, "thigh_hr": 0.7, "calf_hr": -1.4,
        }
        for name, val in default_positions.items():
            addr = self._joint_qpos_addr(name)
            self.data.qpos[addr] = val

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Relative Delta-Aktionen
        for i, name in enumerate(self.joint_names):
            addr = self._joint_qpos_addr(name)
            current_target = self.data.ctrl[i]
            lo, hi = self.joint_ranges[name]
            new_target = current_target + action[i] * self.action_scale[i]
            self.data.ctrl[i] = float(np.clip(new_target, lo, hi))

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        reward, reward_dict = self._compute_reward()
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()
        info.update(reward_dict)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Rumpf Position und Orientierung
        trunk_pos = self.data.xpos[self._ids["trunk"]].copy()
        trunk_quat = self.data.xquat[self._ids["trunk"]].copy()
        trunk_vel = self.data.cvel[self._ids["trunk"]].copy()

        # Gelenke
        joint_pos = []
        joint_vel = []
        for name in self.joint_names:
            addr = self._joint_qpos_addr(name)
            joint_pos.append(self.data.qpos[addr])
            joint_vel.append(self.data.qvel[addr])

        # Fußkontakt (vereinfacht: Höhe des Fußes über Boden)
        feet_contact = []
        for foot in ["foot_fl", "foot_fr", "foot_hl", "foot_hr"]:
            foot_pos = self.data.xpos[self._ids[foot]].copy()
            feet_contact.append(float(foot_pos[2] < 0.03))  # Nah am Boden

        # Zielgeschwindigkeit (vorwärts)
        target_vel = np.array([0.5, 0.0], dtype=np.float32)  # 0.5 m/s vorwärts

        obs = np.concatenate([
            trunk_pos.astype(np.float32),
            trunk_quat.astype(np.float32),
            trunk_vel.astype(np.float32),
            np.array(joint_pos, dtype=np.float32),
            np.array(joint_vel, dtype=np.float32),
            np.array(feet_contact, dtype=np.float32),
            target_vel,
            self.prev_action,
        ])
        return obs

    def _compute_reward(self):
        trunk_pos = self.data.xpos[self._ids["trunk"]]
        trunk_quat = self.data.xquat[self._ids["trunk"]]

        # Vorwärtsgeschwindigkeit (X-Richtung)
        vx = self.data.cvel[self._ids["trunk"]][0]

        # Aufrechte Haltung (Quaternion: w, x, y, z -> Roll/Pitch schätzen)
        # Vereinfacht: prüfe ob Rumpf nicht zu stark geneigt ist
        # trunk_quat[0] = w, nahe 1 = aufrecht
        upright = trunk_quat[0]  # nahe 1.0 = aufrecht

        # Aktuelle X-Position für Distanz
        current_x = trunk_pos[0]
        dx = current_x - self.prev_trunk_x
        self.prev_trunk_x = current_x
        self.episode_distance += dx

        reward = 0.0
        info = {}

        # Belohnung: Vorwärtslaufen
        reward += 5.0 * vx

        # Belohnung: Aufrechte Haltung
        reward += 3.0 * upright

        # Strafe: Umfallen
        if upright < 0.3:
            reward -= 10.0

        # Belohnung: Energieeffizienz (geringe Gelenkgeschwindigkeiten)
        joint_vel_sum = sum(abs(self.data.qvel[self._joint_qpos_addr(n)]) for n in self.joint_names)
        reward -= 0.1 * joint_vel_sum

        # Belohnung: Aktion glätten (wenige Änderungen)
        action_change = np.mean(np.abs(self.prev_action - np.clip(
            self.data.ctrl[:12], -1, 1
        )))
        reward -= 0.5 * action_change

        # Zeitstrafe (fördert schnelles Lernen)
        reward -= 0.02

        info["vx"] = float(vx)
        info["upright"] = float(upright)
        info["distance"] = float(self.episode_distance)
        info["fallen"] = bool(upright < 0.3)

        self.prev_action = np.clip(self.data.ctrl[:12], -1, 1).copy()

        return float(reward), info

    def _is_fallen(self):
        trunk_quat = self.data.xquat[self._ids["trunk"]]
        # Wenn der Rumpf zu stark geneigt ist (w < 0.3), ist er umgefallen
        return bool(trunk_quat[0] < 0.3)

    def _get_info(self):
        return {"step": self.step_count, "distance": self.episode_distance}

    def render(self):
        if self._viewer is None and self.render_mode == "human":
            self._viewer = mj_viewer.launch_passive(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    env = QuadrupedEnv(render_mode=None)
    obs, info = env.reset()
    print("Obs-Dim:", obs.shape)
    print("Action-Dim:", env.action_space.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    print("Smoke-Test V3 OK")
    env.close()
