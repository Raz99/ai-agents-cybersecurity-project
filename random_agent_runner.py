"""
Random agent module and runner utilities.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from cyber_env import CyberKillChainEnv


class RandomAgent:
    """Baseline agent that selects actions uniformly at random."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:
        """Return random action; state is unused by design."""
        # Baseline intentionally ignores state to represent no learning.
        _ = state
        return random.randrange(self.n_actions)


def run_random_agent(
    env: CyberKillChainEnv, episodes: int
) -> Tuple[List[float], List[int], Dict[str, float]]:
    """Run random policy baseline and collect rewards and steps per episode."""
    agent = RandomAgent(env.n_actions)
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    wrong_actions_total = 0
    failed_episodes = 0

    # Each episode is independent and starts from state 0.
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # Roll out until terminal state or max-step cutoff in env.
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if bool(info.get("wrong_action", False)):
                wrong_actions_total += 1
            # Accumulate undiscounted episodic return.
            total_reward += reward
            steps += 1
            state = next_state

        # Store episode-level metrics for downstream comparison/plotting.
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        if state != env.terminal_state:
            failed_episodes += 1

    stats = {
        # Keep this structure consistent with DQN stats for easy comparison output.
        "wrong_actions_total": float(wrong_actions_total),
        "wrong_actions_per_episode": float(wrong_actions_total) / float(episodes),
        "failed_episodes": float(failed_episodes),
        "failure_rate": float(failed_episodes) / float(episodes),
    }
    return episode_rewards, episode_steps, stats
