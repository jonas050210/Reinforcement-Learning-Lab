"""
Gymnasium Environment V2: 3-Achsen-Roboterarm mit 3 Werkzeugen, 3 Objekten und Hindernissen.

Aufgabe: Jedes Objekt mit dem RICHTIGEN Werkzeug greifen und in die passende Box legen.

Werkzeuge (automatisch wechselbar an der Werkzeug-Plattform):
  0 = Parallel-Greifer  -> für Würfel (orange, hartkantig)
  1 = Sauger/Magnet     -> für Kugel (blau, glatt/rund)
  2 = Schaufel          -> für kleinen Teil (lila, klein)

Objekte & Zielboxen sind farblich passend.

Hindernisse müssen vom Arm umfahren werden.

Curriculum: Schwierigkeit steigt automatisch (mehr Objekte, weitere Startpositionen).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer as mj_viewer


class WheelArm3ToolsEnv(gym.Env):
    """Arm mit 3 Gelenken + fahrbarer Basis + 3 Werkzeuge + 3 Objekte + Hindernisse."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Werkzeug-IDs
    TOOL_GRIPPER = 0
    TOOL_SUCTION = 1
    TOOL_SCOOP = 2
    TOOL_NAMES = {0: "greifer", 1: "sauger", 2: "schaufel"}

    # Objekt-Definitionen: (body_name, box_center_site, required_tool, color_rgba, mass)
    OBJECTS = [
        ("cube1", "box_cube_center",   TOOL_GRIPPER, np.array([1.0, 0.5, 0.0, 1.0]), 0.15),
        ("sphere1", "box_sphere_center", TOOL_SUCTION, np.array([0.2, 0.6, 1.0, 1.0]), 0.12),
        ("small1", "box_small_center", TOOL_SCOOP,     np.array([0.7, 0.2, 0.8, 1.0]), 0.05),
    ]

    def __init__(self, xml_path="mujoco_arm_v2.xml", render_mode=None,
                 max_episode_steps=1000, curriculum_stage=0, device="cpu"):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage = curriculum_stage
        self.device = device

        # MuJoCo Modell laden
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # Gelenk-Konfiguration
        self.joint_names = ["base_x", "base_y", "j1", "j2", "j3"]
        self.joint_ranges = {
            "base_x": (-3.0, 3.0),
            "base_y": (-3.0, 3.0),
            "j1": (-2.0, 2.0),
            "j2": (-2.0, 2.0),
            "j3": (-2.0, 2.0),
        }
        self.action_scale = np.array([0.03, 0.03, 0.06, 0.06, 0.06], dtype=np.float32)

        # IDs nachschlagen
        self._ids = self._get_ids()

        # === Action Space ===
        # [dbase_x, dbase_y, dj1, dj2, dj3, tool_select, tool_action]
        # tool_select: 0,1,2 -> Greifer, Sauger, Schaufel
        # tool_action: -1..1 -> Greifer öffnen/schließen (nur Greifer relevant)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 2, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # === Observation Space ===
        # arm_qpos (5) + arm_qvel (5) + current_tool one-hot (3) +
        # pro Objekt: pos (3) + held (1) + in_box (1) = 5 × 3 = 15 +
        # tool_pos (3) +
        # dist zu nächstem Zielobjekt (1) + dist zu dessen Box (1) +
        # obstacle_warning (1) = wie nah am Hindernis
        obs_dim = 5 + 5 + 3 + 15 + 3 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Zustand
        self.step_count = 0
        self.current_tool = self.TOOL_GRIPPER
        self.tool_gripper_open = True
        self.objects_held = [False] * len(self.OBJECTS)
        self.objects_in_box = [False] * len(self.OBJECTS)
        self.prev_objects_in_box = [False] * len(self.OBJECTS)  # für Belohnung nur beim Eintritt
        self.prev_dist_to_target = None
        self.prev_dist_to_box = None
        self.episode_successes = 0

        # Curriculum: wie viele Objekte aktiv sind
        self.n_active_objects = min(1 + curriculum_stage, len(self.OBJECTS))

    # ------------------------------------------------------------------ #
    # IDs
    # ------------------------------------------------------------------ #
    def _get_ids(self):
        ids = {}
        for name in self.joint_names:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        ids["gripper_slide"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_slide"
        )
        for name in ["gripper", "cube1", "sphere1", "small1", "obstacle1", "obstacle2", "obstacle3"]:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        ids["gripper_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip"
        )
        ids["suction_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_tip"
        )
        ids["scoop_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "scoop_tip"
        )
        for _, box_site, _, _, _ in self.OBJECTS:
            ids[box_site] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, box_site
            )
        # Werkzeug-Plattform (zum Wechseln)
        ids["tool_rack"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tool_rack"
        )
        return ids

    # ------------------------------------------------------------------ #
    # Hilfsfunktionen
    # ------------------------------------------------------------------ #
    def _joint_qpos_addr(self, name):
        return int(self.model.jnt_qposadr[self._ids[name]])

    def _object_qpos_addr(self, body_name):
        joint_name = body_name + "_joint"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[jid])

    def _active_tool_site(self):
        if self.current_tool == self.TOOL_GRIPPER:
            return self._ids["gripper_tip"]
        elif self.current_tool == self.TOOL_SUCTION:
            return self._ids["suction_tip"]
        else:
            return self._ids["scoop_tip"]

    def _get_tool_pos(self):
        site_id = self._active_tool_site()
        return self.data.site_xpos[site_id].copy()

    def _get_object_pos(self, obj_idx):
        body_name = self.OBJECTS[obj_idx][0]
        addr = self._object_qpos_addr(body_name)
        return self.data.qpos[addr : addr + 3].copy()

    def _get_box_pos(self, obj_idx):
        box_site = self.OBJECTS[obj_idx][1]
        return self.data.site_xpos[self._ids[box_site]].copy()

    def _get_obstacle_pos(self, name):
        return self.data.xpos[self._ids[name]].copy()

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

    def _dist_to_obstacles(self, pos):
        """Minimale Distanz zu allen Hindernissen (für Warnung/Strafe)."""
        min_dist = 999.0
        for obs_name in ["obstacle1", "obstacle2", "obstacle3"]:
            obs_pos = self._get_obstacle_pos(obs_name)
            dist = np.linalg.norm(pos - obs_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.current_tool = self.TOOL_GRIPPER
        self.tool_gripper_open = True
        self.objects_held = [False] * len(self.OBJECTS)
        self.objects_in_box = [False] * len(self.OBJECTS)
        self.prev_objects_in_box = [False] * len(self.OBJECTS)
        self.prev_dist_to_target = None
        self.prev_dist_to_box = None
        self.episode_successes = 0

        # Arm in Home-Position
        for name in self.joint_names:
            addr = self._joint_qpos_addr(name)
            self.data.qpos[addr] = 0.0
        self.data.qpos[self._joint_qpos_addr("gripper_slide")] = 0.04  # Greifer offen

        mujoco.mj_forward(self.model, self.data)

        # Greifer-Tip Position nach Home-Setup
        gripper_pos = self.data.site_xpos[self._ids["gripper_tip"]].copy()

        # Objekte platzieren (je nach Curriculum-Stufe)
        for obj_idx in range(self.n_active_objects):
            body_name = self.OBJECTS[obj_idx][0]
            addr = self._object_qpos_addr(body_name)

            if self.curriculum_stage == 0:
                # Ein Objekt, direkt vor dem Greifer
                obj_pos = gripper_pos + np.array([0.0, 0.0, 0.02])
            elif self.curriculum_stage == 1:
                # 2 Objekte, leicht versetzt
                obj_pos = gripper_pos + np.array([
                    self.np_random.uniform(-0.1, 0.1),
                    self.np_random.uniform(-0.1, 0.1),
                    0.02
                ])
            elif self.curriculum_stage == 2:
                # 3 Objekte, zufällig in Reichweite
                obj_pos = np.array([
                    self.np_random.uniform(0.3, 1.2),
                    self.np_random.uniform(-0.6, 0.6),
                    0.025
                ])
            else:
                # Schwer: komplett zufällig in der Welt
                obj_pos = np.array([
                    self.np_random.uniform(0.5, 2.0),
                    self.np_random.uniform(-1.5, 0.0),
                    0.025
                ])

            z = obj_pos[2]
            if "cube" in body_name:
                z = 0.03
            elif "sphere" in body_name:
                z = 0.03
            else:
                z = 0.015

            self.data.qpos[addr : addr + 7] = np.array([obj_pos[0], obj_pos[1], z, 1, 0, 0, 0])

        # Nicht-aktive Objekte aus dem Weg räumen
        for obj_idx in range(self.n_active_objects, len(self.OBJECTS)):
            body_name = self.OBJECTS[obj_idx][0]
            addr = self._object_qpos_addr(body_name)
            self.data.qpos[addr : addr + 7] = np.array([10, 10, 10, 1, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # === 1. Gelenk-Aktionen (relative Delta) ===
        for i, name in enumerate(self.joint_names):
            addr = self._joint_qpos_addr(name)
            current_target = self.data.ctrl[i]
            lo, hi = self.joint_ranges[name]
            new_target = current_target + action[i] * self.action_scale[i]
            self.data.ctrl[i] = float(np.clip(new_target, lo, hi))

        # === 2. Werkzeug-Auswahl ===
        new_tool = int(np.clip(round(action[5]), 0, 2))
        if new_tool != self.current_tool:
            self.current_tool = new_tool
            # Werkzeugwechsel: Greifer öffnen
            if self.current_tool == self.TOOL_GRIPPER:
                self.tool_gripper_open = True
                self.data.qpos[self._joint_qpos_addr("gripper_slide")] = 0.04

        # === 3. Werkzeug-Aktion (Greifer öffnen/schließen) ===
        tool_action = action[6]
        if self.current_tool == self.TOOL_GRIPPER:
            if tool_action < -0.5 and self.tool_gripper_open:
                self.tool_gripper_open = False
                self.data.qpos[self._joint_qpos_addr("gripper_slide")] = -0.04
            elif tool_action > 0.5 and not self.tool_gripper_open:
                self.tool_gripper_open = True
                self.data.qpos[self._joint_qpos_addr("gripper_slide")] = 0.04

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # === 4. Objekt-Zustand aktualisieren ===
        self._update_object_states()

        obs = self._get_obs()
        reward, reward_dict = self._compute_reward()
        terminated = self._all_objects_in_box()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()
        info.update(reward_dict)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Objekt-Zustand
    # ------------------------------------------------------------------ #
    def _update_object_states(self):
        tool_pos = self._get_tool_pos()

        for obj_idx in range(self.n_active_objects):
            body_name = self.OBJECTS[obj_idx][0]
            required_tool = self.OBJECTS[obj_idx][2]
            obj_pos = self._get_object_pos(obj_idx)

            # Held: Objekt nah am aktiven Werkzeug UND richtiges Werkzeug
            dist_to_tool = np.linalg.norm(obj_pos - tool_pos)
            is_correct_tool = (self.current_tool == required_tool)

            if dist_to_tool < 0.05 and is_correct_tool:
                if self.current_tool == self.TOOL_GRIPPER:
                    self.objects_held[obj_idx] = not self.tool_gripper_open
                else:
                    self.objects_held[obj_idx] = True
            elif not is_correct_tool:
                self.objects_held[obj_idx] = False

            # In Box: Objekt nah an Box-Zentrale
            box_pos = self._get_box_pos(obj_idx)
            dist_to_box = np.linalg.norm(obj_pos - box_pos)
            if dist_to_box < 0.15 and obj_pos[2] < 0.06:
                self.objects_in_box[obj_idx] = True
            else:
                if not self.objects_held[obj_idx]:
                    self.objects_in_box[obj_idx] = False

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

        # Aktives Werkzeug (one-hot)
        tool_onehot = np.zeros(3, dtype=np.float32)
        tool_onehot[self.current_tool] = 1.0

        # Objekte
        obj_data = []
        for obj_idx in range(len(self.OBJECTS)):
            if obj_idx < self.n_active_objects:
                obj_pos = self._get_object_pos(obj_idx)
                held = float(self.objects_held[obj_idx])
                in_box = float(self.objects_in_box[obj_idx])
            else:
                obj_pos = np.array([10, 10, 10], dtype=np.float32)
                held = 0.0
                in_box = 0.0
            obj_data.extend([obj_pos, np.array([held]), np.array([in_box])])

        # Werkzeug-Position
        tool_pos = self._get_tool_pos()

        # Distanzen zum nächsten NICHT-in-Box Objekt
        target_idx = self._next_target_object()
        if target_idx is not None:
            obj_pos = self._get_object_pos(target_idx)
            box_pos = self._get_box_pos(target_idx)
            dist_to_obj = float(np.linalg.norm(tool_pos - obj_pos))
            dist_to_box = float(np.linalg.norm(obj_pos - box_pos))
        else:
            dist_to_obj = 0.0
            dist_to_box = 0.0

        # Warnung: Nähe zu Hindernissen
        obstacle_dist = self._dist_to_obstacles(tool_pos)
        obstacle_warning = 1.0 / (obstacle_dist + 0.1)  # höher = näher am Hindernis

        obs = np.concatenate([
            np.array(qpos, dtype=np.float32),
            np.array(qvel, dtype=np.float32),
            tool_onehot,
            *obj_data,
            tool_pos.astype(np.float32),
            np.array([dist_to_obj], dtype=np.float32),
            np.array([dist_to_box], dtype=np.float32),
            np.array([obstacle_warning], dtype=np.float32),
        ])
        return obs

    def _next_target_object(self):
        for obj_idx in range(self.n_active_objects):
            if not self.objects_in_box[obj_idx]:
                return obj_idx
        return None

    # ------------------------------------------------------------------ #
    # Belohnung
    # ------------------------------------------------------------------ #
    def _compute_reward(self):
        tool_pos = self._get_tool_pos()
        target_idx = self._next_target_object()

        reward = 0.0
        info = {}

        if target_idx is not None:
            obj_pos = self._get_object_pos(target_idx)
            box_pos = self._get_box_pos(target_idx)
            required_tool = self.OBJECTS[target_idx][2]

            dist_to_obj = float(np.linalg.norm(tool_pos - obj_pos))
            dist_to_box = float(np.linalg.norm(obj_pos - box_pos))

            # Reward Shaping: Annäherung an Zielobjekt
            if self.prev_dist_to_target is not None:
                delta = self.prev_dist_to_target - dist_to_obj
                reward += 5.0 * delta
            self.prev_dist_to_target = dist_to_obj

            # Werkzeug-Auswahl Belohnung
            if self.current_tool == required_tool:
                reward += 0.1
            else:
                reward -= 0.05

            # Hindernis-Strafe (wenn zu nah)
            obstacle_dist = self._dist_to_obstacles(tool_pos)
            if obstacle_dist < 0.15:
                reward -= 2.0 * (0.15 - obstacle_dist)

            # Greifen
            if self.objects_held[target_idx]:
                reward += 3.0
                info["grasped"] = 1.0
            else:
                info["grasped"] = 0.0

            # Zur Box bewegen (nur wenn gehalten)
            if self.objects_held[target_idx]:
                if self.prev_dist_to_box is not None:
                    delta_box = self.prev_dist_to_box - dist_to_box
                    reward += 8.0 * delta_box
                self.prev_dist_to_box = dist_to_box
            else:
                # Zurücksetzen wenn nicht mehr gehalten, damit kein falscher Delta beim nächsten Greifen entsteht
                self.prev_dist_to_box = None

            # In Box (hohe Belohnung nur beim erstmaligen Eintritt, nicht jeden Schritt)
            box_just_entered = (
                self.objects_in_box[target_idx]
                and not self.prev_objects_in_box[target_idx]
            )
            if box_just_entered:
                reward += 50.0
                self.episode_successes += 1
                info["in_box"] = 1.0
            elif self.objects_in_box[target_idx]:
                info["in_box"] = 1.0  # bleibt in Box, aber keine Zusatz-Belohnung
            else:
                info["in_box"] = 0.0
        else:
            dist_to_obj = 0.0
            dist_to_box = 0.0
            info["grasped"] = 0.0
            info["in_box"] = 0.0

        # Zeitstrafe
        reward -= 0.05

        info["dist_to_obj"] = dist_to_obj
        info["dist_to_box"] = dist_to_box
        info["current_tool"] = self.current_tool
        info["objects_in_box"] = sum(self.objects_in_box)
        info["n_active"] = self.n_active_objects

        # Vorherigen Box-Zustand merken für nächste Belohnungs-Berechnung
        self.prev_objects_in_box = self.objects_in_box.copy()

        return float(reward), info

    def _all_objects_in_box(self):
        return all(self.objects_in_box[: self.n_active_objects])

    def _get_info(self):
        return {
            "step": self.step_count,
            "tool": self.current_tool,
            "objects_in_box": sum(self.objects_in_box),
            "n_active": self.n_active_objects,
        }

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
    env = WheelArm3ToolsEnv(render_mode=None, curriculum_stage=0)
    obs, info = env.reset()
    print("Obs-Dim:", obs.shape)
    print("Action-Dim:", env.action_space.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    print("Smoke-Test V2 OK")
    env.close()
