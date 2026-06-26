from stable_baselines3 import PPO
from drone_env import DroneEnv

env = DroneEnv(max_steps=300)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)

print("Начинаем обучение...")
model.learn(total_timesteps=50_000)  # нужно больше времени
model.save("ppo_drone_simple")
print("Модель сохранена")