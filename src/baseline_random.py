import os
import numpy as np
import pandas as pd
import gymnasium as gym


def run_random(env_id="Pendulum-v1", episodes=50, seed=0):
    env = gym.make(env_id)
    rewards = []

    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            # Sample random action
            a = env.action_space.sample()
            obs, r, done, truncated, _ = env.step(a)
            ep_return += float(r)

        rewards.append(ep_return)

    env.close()
    return rewards


def main():
    os.makedirs("results", exist_ok=True)
    rewards = run_random(episodes=50)
    df = pd.DataFrame({"episode": range(1, 51), "return": rewards})
    df.to_csv("results/random_baseline.csv", index=False)
    print("Saved results/random_baseline.csv")


if __name__ == "__main__":
    main()
