from stable_baselines3 import PPO
from drone_env import DroneEnv
import mujoco.viewer
import numpy as np
import time

def fly_waypoints(waypoints, model_path="ppo_drone_straight"):
    """
    Пролёт по заданным точкам.
    
    Args:
        waypoints: список точек [[x1,y1,z1], [x2,y2,z2], ...]
    """
    # Если передана одна точка как плоский список — конвертируем
    if isinstance(waypoints[0], (int, float)):
        waypoints = [waypoints]
    
    env = DroneEnv(max_steps=300)
    model = PPO.load(model_path)
    
    obs, info = env.reset()
    start_pos = env._pos().copy()
    
    current_wp_idx = 0
    env.goal = np.array(waypoints[current_wp_idx])
    env.prev_distance = np.linalg.norm(env._pos() - env.goal)
    
    print("=" * 60)
    print(f" МАРШРУТ ПОЛЁТА ({len(waypoints)} точек)")
    print("=" * 60)
    print(f" Старт: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
    for i, wp in enumerate(waypoints):
        print(f"   Точка {i+1}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
    print("=" * 60)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        
        step = 0
        total_reward = 0
        
        while viewer.is_running() and current_wp_idx < len(waypoints):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            pos = env._pos()
            dist = info['distance']
            
            if step % 20 == 0:
                print(f"Step {step:4d}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
                      f"dist={dist:.2f} → точка {current_wp_idx+1}/{len(waypoints)}")
            
            #  ПРОВЕРКА ДОСТИЖЕНИЯ ТОЧКИ
            if dist < 0.15:
                print(f"   Точка {current_wp_idx+1} достигнута за {step} шагов!")
                current_wp_idx += 1
                
                if current_wp_idx < len(waypoints):
                    env.goal = np.array(waypoints[current_wp_idx])
                    env.prev_distance = np.linalg.norm(pos - env.goal)
                    print(f"   Следующая цель: ({env.goal[0]:.2f}, {env.goal[1]:.2f}, {env.goal[2]:.2f})")
                else:
                    print("\n ВСЕ ТОЧКИ ДОСТИГНУТЫ!")
                    print(f"Суммарная награда: {total_reward:.1f}")
                    break
            
            viewer.sync()
            time.sleep(0.02)
            step += 1
            
            if terminated or truncated:
                print(f" Остановка! Достигнуто {current_wp_idx}/{len(waypoints)} точек")
                break
    
    env.close()
    return current_wp_idx

if __name__ == "__main__":
    
    waypoints = [
        [0.0, 0.0, 1.0],
        [0.7, 0.3, 1.1],
        [1.3, 0.5, 1.5],
        [1.6, 0.5, 1.6],
        [2.3, 0.6, 1.7],
        [2.9, 0.6, 1.6],
        [3.4, 0.4, 1.6],
        [4.0, 0.0, 1.5],
    ]
    
    print(f"Всего точек: {len(waypoints)}")
    fly_waypoints(waypoints, model_path="ppo_drone")