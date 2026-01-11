import os
import yaml
import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from pathlib import Path

# Project root directory (one level above src/)
BASE_DIR = Path(__file__).resolve().parent.parent
# CONFIG_PATH = BASE_DIR / "configs" / "base.yaml"
CONFIG_PATH = BASE_DIR / "pendulum-sac" / "configs" / "base.yaml"





def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main():
    # cfg = load_config("configs/base.yaml")
    cfg = load_config(CONFIG_PATH)
    env_id = cfg["env_id"]
    seed = int(cfg["seed"])

    set_random_seed(seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_env = make_env(env_id, seed)
    eval_env = make_env(env_id, seed + 1)

    # Eval callback writes best model + evaluation metrics to disk
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="results",
        eval_freq=int(cfg["train"]["eval_freq"]),
        n_eval_episodes=int(cfg["train"]["n_eval_episodes"]),
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=float(cfg["sac"]["learning_rate"]),
        buffer_size=int(cfg["sac"]["buffer_size"]),
        batch_size=int(cfg["sac"]["batch_size"]),
        tau=float(cfg["sac"]["tau"]),
        gamma=float(cfg["sac"]["gamma"]),
        train_freq=int(cfg["sac"]["train_freq"]),
        gradient_steps=int(cfg["sac"]["gradient_steps"]),
        ent_coef=cfg["sac"]["ent_coef"],
    )

    total_timesteps = int(cfg["train"]["total_timesteps"])
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=int(cfg["train"]["log_interval"]),
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("models/sac_pendulum_final")
    print("Saved final model to models/sac_pendulum_final.zip")


if __name__ == "__main__":
    main()
