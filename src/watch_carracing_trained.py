# import gymnasium as gym
# from stable_baselines3 import SAC

# # Prefer best_model if it exists
# model_path = "models/best_model.zip"

# env = gym.make("CarRacing-v3", render_mode="human")
# model = SAC.load(model_path)

# obs, _ = env.reset(seed=0)

# for _ in range(2000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         obs, _ = env.reset()

# env.close()


import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack


MODEL_PATH = "models/sac_carracing_final"
ENV_ID = "CarRacing-v3"
SEED = 42


def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env.reset(seed=seed)
        return env
    return _init


# --- Create env EXACTLY like training ---
env = DummyVecEnv([make_env(SEED)])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

# --- Load model WITH env ---
model = SAC.load(MODEL_PATH, env=env)

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
