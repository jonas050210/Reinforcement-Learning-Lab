"""Gymnasium Environment V1-Optimiert: 3-Achsen-Roboterarm auf fahrbarer Basis.

Für Ubuntu + GPU (RTX 4060 Ti) optimiert.
Aufgabe: Fahre zum Würfel, greif ihn, hebe ihn und leg ihn in die Zielbox.

Features:
- Relative Delta-Aktionen (stabileres Lernen)
- Dichte Belohnung (Reward Shaping) für jeden Lernschritt
- Automatisches Curriculum (Würfel startet immer weiter weg)
- GPU-ready (kommt mit device-Parameter, läuft auf CPU oder CUDA)
- Echte MuJoCo-Visualisierung für Desktop-Fenster
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer as mj_viewer


class WheelArmEnv(gym.Env):
    """Ein Arm mit 3 Gelenken + 2 Achsen fahrbare Basis (Räder)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="mujoco_arm.xml", render_mode=None,
                 max_episode_steps=600, curriculum_stage=0, device="cpu"):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage = curriculum_stage
        self.device = device  # "cpu" oder "cuda" — für spätere GPU-Erweiterung

        # MuJoCo Modell laden
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # Gelenk-Konfiguration (muss vor _get_ids stehen)
        self.joint_names = ["base_x", "base_y", "j1", "j2", "j3"]
        self.joint_ranges = {
            "base_x": (-2.5, 2.5),
            "base_y": (-2.5, 2.5),
            "j1": (-2.0, 2.0),
            "j2": (-2.0, 2.0),
            "j3": (-2.0, 2.0),
        }
        # Max Delta pro Schritt pro Gelenk
        self.action_scale = np.array([0.03, 0.03, 0.06, 0.06, 0.06], dtype=np.float32)

        # IDs nachschlagen
        self._ids = self._get_ids()

        # Action Space: relative Änderungen [dbase_x, dbase_y, dj1, dj2, dj3]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space:
        #   arm_qpos (5) + arm_qvel (5) + cube_pos (3) + gripper_pos (3)
        #   cube_to_gripper (3) + cube_to_box (3) + prev_dist_gc (1) + prev_dist_cb (1)
        obs_dim = 5 + 5 + 3 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Zustand
        self.step_count = 0
        self.prev_dist_gripper_cube = None
        self.prev_dist_cube_box = None
        self.lifted_bonus_given = False
        self.episode_successes = 0

    # ------------------------------------------------------------------ #
    # IDs
    # ------------------------------------------------------------------ #
    def _get_ids(self):
        ids = {}
        for name in self.joint_names:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        for name in ["gripper", "cube", "target_box"]:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        ids["gripper_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip"
        )
        ids["target_center"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_center"
        )
        return ids

    # ------------------------------------------------------------------ #
    # Hilfsfunktionen
    # ------------------------------------------------------------------ #
    def _joint_qpos_addr(self, name):
        return int(self.model.jnt_qposadr[self._ids[name]])

    def _cube_qpos_addr(self):
        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        return int(self.model.jnt_qposadr[cube_jid])

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_dist_gripper_cube = None
        self.prev_dist_cube_box = None
        self.lifted_bonus_given = False
        self.episode_successes = 0

        # Arm in Home-Position (alle Gelenke auf 0)
        for name in self.joint_names:
            addr = self._joint_qpos_addr(name)
            self.data.qpos[addr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Greifer-Tip Position nach Home-Setup
        gripper_pos = self.data.site_xpos[self._ids["gripper_tip"]].copy()

        # Würfel-Startposition je nach Curriculum-Stufe
        if self.curriculum_stage == 0:
            # Direkt vor dem Greifer (einfach)
            cube_pos = gripper_pos + np.array([0.0, 0.0, 0.02])
        elif self.curriculum_stage == 1:
            # Leicht versetzt
            cube_pos = gripper_pos + np.array([
                self.np_random.uniform(-0.08, 0.08),
                self.np_random.uniform(-0.08, 0.08),
                0.02
            ])
        elif self.curriculum_stage == 2:
            # Zufällig in Reichweite
            cube_pos = np.array([
                self.np_random.uniform(0.3, 0.9),
                self.np_random.uniform(-0.3, 0.6),
                0.025
            ])
        else:
            # Schwer: komplett zufällig in der Welt
            cube_pos = np.array([
                self.np_random.uniform(0.5, 1.5),
                self.np_random.uniform(-1.0, 0.0),
                0.025
            ])

        cube_addr = self._cube_qpos_addr()
        self.data.qpos[cube_addr : cube_addr + 7] = np.array(
            [cube_pos[0], cube_pos[1], cube_pos[2], 1, 0, 0, 0]
        )

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Relative Delta-Aktionen auf die aktuellen Gelenkzielpositionen
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
        terminated = self._is_success()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()
        info.update(reward_dict)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Beobachtung
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        qpos = []
        qvel = []
        for name in self.joint_names:
            addr = self._joint_qpos_addr(name)
            qpos.append(self.data.qpos[addr])
            qvel.append(self.data.qvel[addr])

        cube_addr = self._cube_qpos_addr()
        cube_pos = self.data.qpos[cube_addr : cube_addr + 3]

        gripper_pos = self.data.site_xpos[self._ids["gripper_tip"]]
        box_center = self.data.site_xpos[self._ids["target_center"]]

        cube_to_gripper = gripper_pos - cube_pos
        cube_to_box = box_center - cube_pos

        d_gc = float(np.linalg.norm(gripper_pos - cube_pos))
        d_cb = float(np.linalg.norm(cube_pos - box_center))

        obs = np.concatenate([
            np.array(qpos, dtype=np.float32),
            np.array(qvel, dtype=np.float32),
            cube_pos.astype(np.float32),
            gripper_pos.astype(np.float32),
            cube_to_gripper.astype(np.float32),
            cube_to_box.astype(np.float32),
            np.array([d_gc], dtype=np.float32),
            np.array([d_cb], dtype=np.float32),
        ])
        return obs

    # ------------------------------------------------------------------ #
    # Belohnung
    # ------------------------------------------------------------------ #
    def _compute_reward(self):
        gripper_pos = self.data.site_xpos[self._ids["gripper_tip"]]
        cube_addr = self._cube_qpos_addr()
        cube_pos = self.data.qpos[cube_addr : cube_addr + 3]
        box_center = self.data.site_xpos[self._ids["target_center"]]

        d_gc = float(np.linalg.norm(gripper_pos - cube_pos))
        d_cb = float(np.linalg.norm(cube_pos - box_center))

        reward = 0.0
        info = {}

        # --- Reward Shaping: Fortschritt gegenüber letzten Schritt ---
        if self.prev_dist_gripper_cube is not None:
            delta_gc = self.prev_dist_gripper_cube - d_gc
            reward += 5.0 * delta_gc  # Belohnung für Annäherung an Würfel
        if self.prev_dist_cube_box is not None:
            delta_cb = self.prev_dist_cube_box - d_cb
            reward += 10.0 * delta_cb  # Starke Belohnung für Würfel→Box

        self.prev_dist_gripper_cube = d_gc
        self.prev_dist_cube_box = d_cb

        # Zeitstrafe pro Schritt (fördert schnelles Lösen)
        reward -= 0.1

        # Kontakt: Greifer berührt Würfel
        touching = self._bodies_touching("gripper", "cube")
        if touching:
            reward += 2.0
            info["touching"] = 1.0
        else:
            info["touching"] = 0.0

        # Würfel gehoben (nur einmalig belohnen)
        if cube_pos[2] > 0.05 and not self.lifted_bonus_given:
            reward += 5.0
            self.lifted_bonus_given = True
            info["lifted"] = 1.0
        else:
            info["lifted"] = 0.0

        # Erfolg: Würfel in Box (hohe Belohnung)
        if d_cb < 0.15 and cube_pos[2] < 0.06:
            reward += 100.0
            info["in_box"] = 1.0
            self.episode_successes += 1
        else:
            info["in_box"] = 0.0

        info["dist_gripper_cube"] = d_gc
        info["dist_cube_box"] = d_cb

        return float(reward), info

    def _is_success(self):
        cube_addr = self._cube_qpos_addr()
        cube_pos = self.data.qpos[cube_addr : cube_addr + 3]
        box_center = self.data.site_xpos[self._ids["target_center"]]
        return bool(np.linalg.norm(cube_pos - box_center) < 0.15 and cube_pos[2] < 0.06)

    def _bodies_touching(self, body_a: str, body_b: str) -> bool:
        body_id_a = self._ids[body_a]
        body_id_b = self._ids[body_b]
        geom_ids_a = set()
        geom_ids_b = set()
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == body_id_a:
                geom_ids_a.add(g)
            if self.model.geom_bodyid[g] == body_id_b:
                geom_ids_b.add(g)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in geom_ids_a and c.geom2 in geom_ids_b) or \
               (c.geom1 in geom_ids_b and c.geom2 in geom_ids_a):
                return True
        return False

    def _get_info(self):
        return {"step": self.step_count}

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
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
    env = WheelArmEnv(render_mode=None)
    obs, info = env.reset()
    print("Obs-Dim:", obs.shape)
    print("Action-Dim:", env.action_space.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    print("Smoke-Test OK")
    env.close()
