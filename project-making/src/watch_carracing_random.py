import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human")
obs, _ = env.reset(seed=0)

for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

env.close()
