import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed


def make_env(seed: int, render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init


def main():
    seed = 42
    set_random_seed(seed)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Training env (vectorized)
    train_env = DummyVecEnv([make_env(seed)])
    train_env = VecTransposeImage(train_env)      # (H,W,C) -> (C,H,W)
    train_env = VecFrameStack(train_env, n_stack=4)  # adds temporal context

    # Eval env
    eval_env = DummyVecEnv([make_env(seed + 1)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="results",
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # model = SAC(
    #     policy="CnnPolicy",
    #     env=train_env,
    #     verbose=1,
    #     seed=seed,
    #     tensorboard_log="logs",
    #     device="cuda",  # will fall back to cpu if CUDA is unavailable
    #     learning_rate=3e-4,
    #     buffer_size=200_000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=1,
    #     gradient_steps=1,
    #     ent_coef="auto",
    # )

    model = SAC(
        policy="CnnPolicy",
        env=train_env,

        # ---- MEMORY-CRITICAL ----
        buffer_size=10_000,      # 🔴 was 200k → now safe
        # buffer_size=30_000,      # 🔴 was 200k → now safe
        # batch_size=64,           # small batch
        batch_size=32,           # small batch

        # ---- TRAINING ----
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,

        # ---- RL HYPERPARAMS ----
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",

        # ---- SYSTEM ----
        device="cpu",
        verbose=1,
    )


    # CarRacing is harder. Start with this, then scale up if needed.
    # total_timesteps = 500_000
    total_timesteps = 100_000

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("models/sac_carracing_final")
    print("Saved models/sac_carracing_final.zip")


if __name__ == "__main__":
    main()
