import os
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import SAC


def evaluate(model_path: str, env_id="Pendulum-v1", episodes=20, seed=123):
    # Record video
    os.makedirs("videos", exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos",
        name_prefix="sac_pendulum",
        episode_trigger=lambda ep: ep == 0,  # record first episode only
        disable_logger=True,
    )

    model = SAC.load(model_path)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_return += float(r)

        returns.append(ep_return)

    env.close()
    return returns


def main():
    os.makedirs("results", exist_ok=True)

    # Use best model if EvalCallback saved it
    model_path = "models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "models/sac_pendulum_final.zip"

    returns = evaluate(model_path=model_path, episodes=20)
    df = pd.DataFrame({"episode": range(1, 21), "return": returns})
    df.to_csv("results/sac_eval.csv", index=False)

    print(f"Saved results/sac_eval.csv, mean return = {np.mean(returns):.2f}")
    print("Video should be in videos/ (first episode recorded)")


if __name__ == "__main__":
    main()
