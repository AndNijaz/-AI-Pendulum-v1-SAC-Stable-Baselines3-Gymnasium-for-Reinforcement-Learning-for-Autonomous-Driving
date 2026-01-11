import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("plots", exist_ok=True)

    rand = pd.read_csv("results/random_baseline.csv")
    sac = pd.read_csv("results/sac_eval.csv")

    plt.figure()
    plt.plot(rand["episode"], rand["return"], label="Random policy")
    plt.plot(sac["episode"], sac["return"], label="SAC (eval)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Pendulum-v1: Random baseline vs SAC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/baseline_vs_sac.png", dpi=200)
    plt.close()

    print("Saved plots/baseline_vs_sac.png")


if __name__ == "__main__":
    main()
