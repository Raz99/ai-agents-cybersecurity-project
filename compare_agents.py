"""
Main experiment runner: compare Random agent vs DQN agent.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from cyber_env import CyberKillChainEnv, load_netsec_dataset, set_seed
from dqn_agent_runner import run_dqn_agent
from random_agent_runner import run_random_agent


def moving_average(values: List[float], window_size: int = 50) -> np.ndarray:
    """Compute moving average for smoother visualization."""
    if window_size <= 1:
        return np.array(values, dtype=np.float32)
    if len(values) < window_size:
        return np.array(values, dtype=np.float32)

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    ma_valid = np.convolve(values, kernel, mode="valid")
    # Left-pad so the smoothed curve aligns with episode indexing.
    pad_left = window_size - 1
    pad = np.full((pad_left,), ma_valid[0], dtype=np.float32)
    return np.concatenate([pad, ma_valid])


def plot_rewards(
    random_rewards: List[float],
    dqn_rewards: List[float],
    window_size: int = 50,
    save_path: str = "reward_comparison.png",
) -> None:
    """Generate and save a smoothed reward comparison figure."""
    # Episode index starts at 1 for presentation readability.
    episodes = np.arange(1, len(random_rewards) + 1)
    random_smooth = moving_average(random_rewards, window_size)
    dqn_smooth = moving_average(dqn_rewards, window_size)

    plt.figure(figsize=(10, 6))
    # Plot only smoothed curves for a cleaner academic figure.
    plt.plot(episodes, random_smooth, label="Random Agent (Moving Avg)", linewidth=2)
    plt.plot(episodes, dqn_smooth, label="DQN Agent (Moving Avg)", linewidth=2)
    plt.title("Random Agent vs DQN Agent on Cyber Kill-Chain Environment", fontsize=13)
    plt.xlabel("Episodes", fontsize=11)
    plt.ylabel("Total Reward", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_steps(
    random_steps: List[int],
    dqn_steps: List[int],
    window_size: int = 50,
    save_path: str = "steps_comparison.png",
) -> None:
    """Generate and save a smoothed steps-per-episode comparison figure."""
    # Same x-axis convention as reward plot for direct visual comparison.
    episodes = np.arange(1, len(random_steps) + 1)
    random_smooth = moving_average([float(s) for s in random_steps], window_size)
    dqn_smooth = moving_average([float(s) for s in dqn_steps], window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, random_smooth, label="Random Agent (Moving Avg)", linewidth=2)
    plt.plot(episodes, dqn_smooth, label="DQN Agent (Moving Avg)", linewidth=2)
    plt.title("Steps per Episode: Random Agent vs DQN Agent", fontsize=13)
    plt.xlabel("Episodes", fontsize=11)
    plt.ylabel("Steps per Episode", fontsize=11)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def print_metrics(name: str, rewards: List[float], steps: List[int]) -> None:
    """Print concise summary statistics for an agent."""
    print(f"\n{name} Results")
    print("-" * (len(name) + 8))
    print(f"Episodes: {len(rewards)}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Best Reward: {np.max(rewards):.2f}")
    # Final-100 metric highlights end-of-training policy quality.
    print(f"Final 100-Episode Avg Reward: {np.mean(rewards[-100:]):.2f}")


def print_failure_logs(name: str, stats: Dict[str, float]) -> None:
    """Print wrong-action and failure-case statistics for an agent."""
    print(f"\n{name} Failure Logging")
    print("-" * (len(name) + 16))
    print(f"Wrong Actions (Total): {int(stats['wrong_actions_total'])}")
    print(f"Wrong Actions per Episode: {stats['wrong_actions_per_episode']:.2f}")
    print(f"Failed Episodes: {int(stats['failed_episodes'])}")
    print(f"Failure Rate: {stats['failure_rate'] * 100:.2f}%")


def print_final_comparison(
    random_rewards: List[float],
    random_steps: List[int],
    dqn_rewards: List[float],
    dqn_steps: List[int],
) -> None:
    """Print clear side-by-side final comparison summary."""
    print("\nFinal Comparison")
    print("----------------")
    print("Random Agent:")
    print(f"- Avg Steps: {np.mean(random_steps):.2f}")
    print(f"- Avg Reward: {np.mean(random_rewards):.2f}")
    print(f"- Final 100 Avg Reward: {np.mean(random_rewards[-100:]):.2f}")
    print("\nDQN Agent:")
    print(f"- Avg Steps: {np.mean(dqn_steps):.2f}")
    print(f"- Avg Reward: {np.mean(dqn_rewards):.2f}")
    print(f"- Final 100 Avg Reward: {np.mean(dqn_rewards[-100:]):.2f}")


def main() -> None:
    """Run end-to-end comparison experiment."""
    set_seed(42)
    # PyTorch has its own RNG; seed separately for repeatable DQN runs.
    torch.manual_seed(42)

    episodes = 1000
    smoothing_window = 50
    dataset_path = "data/netsecdata.csv"

    # Load data once and pass the same inferred dynamics to both agents.
    rows, transition_probs = load_netsec_dataset(dataset_path=dataset_path)
    if rows is not None:
        print(f"[Info] Loaded dataset rows: {len(rows)}")
    print(f"[Info] Using transition probabilities: {transition_probs}")

    # Separate env instances keep runs independent and reproducible.
    random_env = CyberKillChainEnv(max_steps=20, transition_probabilities=transition_probs)
    dqn_env = CyberKillChainEnv(max_steps=20, transition_probabilities=transition_probs)

    # Each runner returns episode metrics plus aggregated failure statistics.
    random_rewards, random_steps, random_stats = run_random_agent(random_env, episodes)
    dqn_rewards, dqn_steps, _, dqn_stats = run_dqn_agent(dqn_env, episodes)

    # Console metrics complement the plot with easy-to-quote values.
    print_metrics("Random Agent", random_rewards, random_steps)
    print_metrics("DQN Agent", dqn_rewards, dqn_steps)
    print_failure_logs("Random Agent", random_stats)
    print_failure_logs("DQN Agent", dqn_stats)
    print_final_comparison(random_rewards, random_steps, dqn_rewards, dqn_steps)

    plot_rewards(random_rewards, dqn_rewards, smoothing_window, "reward_comparison.png")
    plot_steps(random_steps, dqn_steps, smoothing_window, "steps_comparison.png")


if __name__ == "__main__":
    main()
