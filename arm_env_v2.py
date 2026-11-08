"""Gymnasium Environment V2: 3-Achsen-Roboterarm mit 3 Werkzeugen und mehreren Objekten.

Aufgabe: Hebe JEDES Objekt mit dem RICHTIGEN Werkzeug auf und bring es zur passenden Box.

Werkzeuge:
  0 = Parallel-Greifer (für Würfel 1, orange)
  1 = Sauger/Magnet    (für Würfel 2, blau)
  2 = Schaufel         (für kleinen Teil, lila)

Objekte & Zielboxen sind farblich passend.

Curriculum: Schwierigkeit steigt automatisch (Würfel starten weiter weg, mehr Objekte).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer as mj_viewer


class WheelArm3ToolsEnv(gym.Env):
    """Arm mit 3 Gelenken + fahrbarer Basis + 3 austauschbare Werkzeuge + 3 Objekte."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    # Werkzeug-IDs
    TOOL_GRIPPER = 0
    TOOL_SUCTION = 1
    TOOL_SCOOP = 2
    TOOL_NAMES = {0: "greifer", 1: "sauger", 2: "schaufel"}

    # Objekt-Definitionen: (body_name, box_center_site, required_tool, color_rgba)
    OBJECTS = [
        ("cube1", "box_orange_center", TOOL_GRIPPER, np.array([1.0, 0.5, 0.0, 1.0])),
        ("cube2", "box_blue_center",   TOOL_SUCTION, np.array([0.2, 0.6, 1.0, 1.0])),
        ("smallpart1", "box_purple_center", TOOL_SCOOP, np.array([0.6, 0.2, 0.8, 1.0])),
    ]

    def __init__(self, xml_path="mujoco_arm_v2.xml", render_mode=None,
                 max_episode_steps=800, curriculum_stage=0):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.curriculum_stage = curriculum_stage

        # MuJoCo Modell
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # Gelenk-Konfiguration (muss vor _get_ids() stehen)
        self.joint_names = ["base_x", "base_y", "j1", "j2", "j3"]
        self.joint_ranges = {
            "base_x": (-2.0, 2.0),
            "base_y": (-2.0, 2.0),
            "j1": (-1.57, 1.57),
            "j2": (-1.57, 1.57),
            "j3": (-1.57, 1.57),
        }
        self.action_scale = np.array([0.02, 0.02, 0.05, 0.05, 0.05], dtype=np.float32)

        # IDs nachschlagen
        self._ids = self._get_ids()

        # === Action Space ===
        # [dbase_x, dbase_y, dj1, dj2, dj3, tool_select]
        # tool_select: -1..1 -> 0,1,2 (Greifer, Sauger, Schaufel)
        # Greifer schließt automatisch bei Berührung (magnetisch)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # === Observation Space ===
        # arm_qpos (5) + arm_qvel (5) + current_tool one-hot (3) +
        # für jedes Objekt: pos (3) + velocity (3) + held (1) + in_box (1) = 8 × 3 = 24 +
        # tool_pos (3) +
        # dist zu nächstem Zielobjekt (1) + dist zu dessen Box (1)
        obs_dim = 5 + 5 + 3 + 8 * len(self.OBJECTS) + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Zustand
        self.step_count = 0
        self.current_tool = self.TOOL_GRIPPER
        self.tool_gripper_open = True  # Greiferzustand
        self.objects_held = [False] * len(self.OBJECTS)
        self.objects_in_box = [False] * len(self.OBJECTS)
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
        # Tool-Joint
        ids["tool_joint"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "tool_joint"
        )
        # Körper
        for name in ["gripper", "cube1", "cube2", "smallpart1", "obstacle1"]:
            ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        # Sites
        ids["gripper_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip"
        )
        ids["suction_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_tip"
        )
        ids["scoop_tip"] = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "scoop_tip"
        )
        for _, box_site, _, _ in self.OBJECTS:
            ids[box_site] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, box_site
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
        """Gibt die Site-ID des aktuell aktiven Werkzeugs zurück."""
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

    def _get_object_vel(self, obj_idx):
        body_name = self.OBJECTS[obj_idx][0]
        addr = self._object_qpos_addr(body_name)
        # qvel Adresse ist nicht gleich qpos Adresse, aber für free joints:
        # qvel hat 6 Werte (3 pos + 3 rot). Wir nehmen die ersten 3.
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                body_name + "_joint")
        vaddr = int(self.model.jnt_dofadr[jid])
        return self.data.qvel[vaddr : vaddr + 3].copy()

    def _get_box_pos(self, obj_idx):
        box_site = self.OBJECTS[obj_idx][1]
        return self.data.site_xpos[self._ids[box_site]].copy()

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
        self.prev_dist_to_target = None
        self.prev_dist_to_box = None
        self.episode_successes = 0

        # Arm in Home-Position
        for name in self.joint_names:
            addr = self._joint_qpos_addr(name)
            self.data.qpos[addr] = 0.0

        # Greifer geöffnet
        self.data.qpos[self._joint_qpos_addr("tool_joint")] = 0.03  # offen

        mujoco.mj_forward(self.model, self.data)

        # Objekte platzieren (je nach Curriculum-Stufe)
        self._place_objects()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _place_objects(self):
        """Platziere aktive Objekte zufällig in Reichweite."""
        for obj_idx in range(self.n_active_objects):
            body_name = self.OBJECTS[obj_idx][0]
            addr = self._object_qpos_addr(body_name)

            # Zufällige Position (je nach Schwierigkeit weiter weg)
            if self.curriculum_stage == 0:
                x = self.np_random.uniform(0.1, 0.5)
                y = self.np_random.uniform(-0.3, 0.3)
            elif self.curriculum_stage == 1:
                x = self.np_random.uniform(0.2, 0.8)
                y = self.np_random.uniform(-0.5, 0.5)
            else:
                x = self.np_random.uniform(0.2, 1.2)
                y = self.np_random.uniform(-0.8, 0.8)

            z = 0.025 if "cube" in body_name else 0.01
            self.data.qpos[addr : addr + 7] = np.array([x, y, z, 1, 0, 0, 0])

            # Nicht-aktive Objekte aus dem Weg räumen (hoch in die Luft)
        for obj_idx in range(self.n_active_objects, len(self.OBJECTS)):
            body_name = self.OBJECTS[obj_idx][0]
            addr = self._object_qpos_addr(body_name)
            self.data.qpos[addr : addr + 7] = np.array([10, 10, 10, 1, 0, 0, 0])

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # === 1. Gelenk-Aktionen (relative Delta) ===
        for i, name in enumerate(self.joint_names):
            addr = self._joint_qpos_addr(name)
            current_target = self.data.ctrl[i]
            lo, hi = self.joint_ranges[name]
            new_target = current_target + action[i] * self.action_scale[i]
            self.data.ctrl[i] = float(np.clip(new_target, lo, hi))

        # === 2. Werkzeug-Auswahl ===
        tool_val = action[5]
        new_tool = int(round((tool_val + 1) / 2 * 2))  # -1..1 -> 0,1,2
        new_tool = np.clip(new_tool, 0, 2)
        if new_tool != self.current_tool:
            self.current_tool = new_tool
            # Werkzeugwechsel: Greifer öffnen beim Wechsel
            if self.current_tool == self.TOOL_GRIPPER:
                self.tool_gripper_open = True
                self.data.qpos[self._joint_qpos_addr("tool_joint")] = 0.03

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # === 4. Objekt-Zustand aktualisieren (gehalten / in Box) ===
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

            # Held: Objekt ist nah am aktiven Werkzeug UND richtiges Werkzeug
            # (Greifer schließt automatisch bei Berührung = magnetisch)
            dist_to_tool = np.linalg.norm(obj_pos - tool_pos)
            is_correct_tool = (self.current_tool == required_tool)

            if dist_to_tool < 0.05 and is_correct_tool:
                self.objects_held[obj_idx] = True
            elif not is_correct_tool:
                # Falsches Werkzeug -> Objekt nicht halten
                self.objects_held[obj_idx] = False

            # In Box: Objektposition nah an Box-Zentrale
            box_pos = self._get_box_pos(obj_idx)
            dist_to_box = np.linalg.norm(obj_pos - box_pos)
            if dist_to_box < 0.12 and obj_pos[2] < 0.06:
                self.objects_in_box[obj_idx] = True
            else:
                # Wenn Objekt nicht mehr in Box (z.B. weggetragen), zurücksetzen
                # nur wenn es nicht gehalten wird
                if not self.objects_held[obj_idx]:
                    self.objects_in_box[obj_idx] = False

    # ------------------------------------------------------------------ #
    # Beobachtung
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        # Arm Gelenke
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
                obj_vel = self._get_object_vel(obj_idx)
                held = float(self.objects_held[obj_idx])
                in_box = float(self.objects_in_box[obj_idx])
            else:
                obj_pos = np.array([10, 10, 10], dtype=np.float32)
                obj_vel = np.zeros(3, dtype=np.float32)
                held = 0.0
                in_box = 0.0
            obj_data.extend([obj_pos, obj_vel, np.array([held]), np.array([in_box])])

        # Werkzeug-Position (aktiv)
        tool_pos = self._get_tool_pos()

        # Distanzen zum nächsten NICHT-in-Box Objekt
        target_obj_idx = self._next_target_object()
        if target_obj_idx is not None:
            obj_pos = self._get_object_pos(target_obj_idx)
            box_pos = self._get_box_pos(target_obj_idx)
            dist_to_obj = float(np.linalg.norm(tool_pos - obj_pos))
            dist_to_box = float(np.linalg.norm(obj_pos - box_pos))
        else:
            dist_to_obj = 0.0
            dist_to_box = 0.0

        obs_parts = [
            np.array(qpos, dtype=np.float32),
            np.array(qvel, dtype=np.float32),
            tool_onehot,
        ]
        for item in obj_data:
            obs_parts.append(np.atleast_1d(item).astype(np.float32))
        obs_parts.append(tool_pos.astype(np.float32))
        obs_parts.append(np.array([dist_to_obj], dtype=np.float32))
        obs_parts.append(np.array([dist_to_box], dtype=np.float32))
        obs = np.concatenate(obs_parts)
        return obs

    def _next_target_object(self):
        """Gibt Index des nächsten Objekts zurück, das noch nicht in der Box ist."""
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
                reward += 0.1  # kleines Plus für richtiges Werkzeug
            else:
                reward -= 0.05  # kleiner Straf für falsches Werkzeug

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
                    reward += 8.0 * delta_box  # Belohnung wenn näher zur Box
                self.prev_dist_to_box = dist_to_box

            # In Box
            if self.objects_in_box[target_idx]:
                reward += 50.0
                info["in_box"] = 1.0
                self.episode_successes += 1
            else:
                info["in_box"] = 0.0

        else:
            # Alle Objekte in Box
            dist_to_obj = 0.0
            dist_to_box = 0.0
            info["grasped"] = 0.0
            info["in_box"] = 0.0

        # Zeitstrafe
        reward -= 0.02

        info["dist_to_obj"] = dist_to_obj if target_idx is not None else 0.0
        info["dist_to_box"] = dist_to_box if target_idx is not None else 0.0
        info["current_tool"] = self.current_tool
        info["objects_in_box"] = sum(self.objects_in_box)
        info["n_active"] = self.n_active_objects

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
    print("Smoke-Test OK")
    env.close()
