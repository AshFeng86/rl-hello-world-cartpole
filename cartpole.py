import gymnasium as gym
from stable_baselines3 import PPO

# Train
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Watch the trained policy
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
env.close()