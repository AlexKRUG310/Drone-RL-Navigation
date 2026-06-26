import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class DroneEnv(gym.Env):
    """Среда для обучения дрона с пролётом по точкам."""

    def __init__(
        self,
        scene_path="scene.xml",
        max_steps=300,
        frame_skip=5,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0

        self.joint = self.model.joint("drone_free")
        self.joint_id = self.joint.id

        self.qpos_addr = self.model.jnt_qposadr[self.joint_id]
        self.qvel_addr = self.model.jnt_dofadr[self.joint_id]

        self.body_id = self.model.body("drone").id
        self.mass = self.model.body_mass[self.body_id]

        self.hover_force = self.mass * 9.81

        print(f"[INFO] mass = {self.mass:.4f} кг")
        print(f"[INFO] hover_force = {self.hover_force:.4f} Н")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self.max_force_xy = 0.8
        self.max_force_z = 0.4
        self.max_speed = 2.0
        self.min_height = 0.1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        self.goal = None
        self.prev_distance = None
        self.start_pos = None

    def _pos(self):
        return self.data.qpos[
            self.qpos_addr:self.qpos_addr + 3
        ].copy()

    def _vel(self):
        return self.data.qvel[
            self.qvel_addr:self.qvel_addr + 3
        ].copy()

    def _obs(self):
        pos = self._pos()
        vel = self._vel()
        delta = self.goal - pos
        dist = np.linalg.norm(delta)
        return np.concatenate([delta, vel, [dist]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        start = np.array([0.0, 0.0, 0.1])
        self.start_pos = start.copy()

        self.data.qpos[self.qpos_addr:self.qpos_addr + 3] = start
        self.data.qpos[self.qpos_addr + 3:self.qpos_addr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.data.qvel[:] = 0.0

        self.goal = np.array([
            np.random.uniform(1.0, 3.5),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(0.5, 1.5),
        ])

        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

        start_dist = np.linalg.norm(self.goal - start)
        self.prev_distance = start_dist

        return self._obs(), {"goal": self.goal.copy()}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        fx = action[0] * self.max_force_xy
        fy = action[1] * self.max_force_xy
        fz = self.hover_force + action[2] * self.max_force_z

        self.data.ctrl[0] = fx
        self.data.ctrl[1] = fy
        self.data.ctrl[2] = fz

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        pos = self._pos()
        vel = self._vel()
        speed = np.linalg.norm(vel)

        delta = self.goal - pos
        dist = np.linalg.norm(delta)

        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.data.qvel[self.qpos_addr:self.qpos_addr + 3] *= scale
            vel = self._vel()
            speed = self.max_speed

        #  НАГРАДА ЗА ПРИБЛИЖЕНИЕ
        progress = self.prev_distance - dist
        reward = 10.0 * progress

        # Маленькая награда за выживание
        reward += 0.01

        # Штраф за скорость
        reward -= 0.005 * speed

        # Штраф за резкие действия
        reward -= 0.001 * np.sum(action ** 2)

        #  ШТРАФ ЗА ВЕРТИКАЛЬНЫЕ ПЕРЕМЕЩЕНИЯ (чтобы летел по прямой)
        vertical_speed = abs(vel[2])
        horizontal_speed = np.linalg.norm(vel[:2])
        reward -= 0.03 * vertical_speed
        reward += 0.01 * horizontal_speed

        #  ШТРАФ ЗА ОТКЛОНЕНИЕ ОТ ПРЯМОЙ
        if self.start_pos is not None:
            direct_vector = self.goal - self.start_pos
            direct_dist = np.linalg.norm(direct_vector)
            if direct_dist > 0.01:
                t = np.dot(pos - self.start_pos, direct_vector) / (direct_dist ** 2)
                t = np.clip(t, 0.0, 1.0)
                projection = self.start_pos + t * direct_vector
                deviation = np.linalg.norm(pos - projection)
                reward -= 0.3 * deviation

        terminated = False
        truncated = False

        #  УСПЕХ (БЕЗ TERMINATED! ТОЛЬКО НАГРАДА)
        if dist < 0.15:
            reward += 200.0
            #  НЕ ставим terminated=True!
            # Сбрасываем prev_distance, чтобы дрон начал движение к следующей цели
            self.prev_distance = 0.0
            print(f"Цель достигнута! dist={dist:.3f}")

        #  ПАДЕНИЕ
        if pos[2] < self.min_height:
            reward -= 50.0
            terminated = True
            print(f"Падение! pos[2]={pos[2]:.2f}")

        #  СЛИШКОМ ВЫСОКО
        if pos[2] > 4.0:
            reward -= 50.0
            terminated = True
            print(f"Слишком высоко! pos[2]={pos[2]:.2f}")

        #  ВЫЛЕТ ЗА ГРАНИЦЫ
        if abs(pos[0]) > 5.0 or abs(pos[1]) > 5.0:
            reward -= 50.0
            terminated = True
            print(f"Вылет за границы! pos=({pos[0]:.2f}, {pos[1]:.2f})")

        if self.step_count >= self.max_steps:
            truncated = True

        self.prev_distance = dist

        info = {
            "distance": dist,
            "goal": self.goal.copy(),
            "speed": speed,
            "position": pos.copy(),
            "height": pos[2],
        }

        return self._obs(), reward, terminated, truncated, info