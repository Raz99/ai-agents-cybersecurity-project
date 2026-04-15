"""
DQN agent module and training runner utilities.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cyber_env import CyberKillChainEnv


class QNetwork(nn.Module):
    """Small feedforward network for DQN."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # Small MLP is enough due to very low-dimensional discrete state space.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    """Single transition tuple for replay memory."""

    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ReplayBuffer:
    """Fixed-size FIFO replay memory for experience replay."""

    def __init__(self, capacity: int = 5000) -> None:
        # FIFO buffer keeps a rolling window of recent experience.
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        # Uniform sampling breaks short-term correlation between transitions.
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent with target network and epsilon-greedy policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 5000,
        target_update_freq: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        # Core hyperparameters for value learning.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cpu")
        # Online net is optimized directly; target net provides stable targets.
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.learn_steps = 0

    def _one_hot(self, state_indices: np.ndarray) -> torch.Tensor:
        # State space is tiny/discrete, so one-hot encoding is sufficient.
        states = torch.zeros((len(state_indices), self.state_dim), dtype=torch.float32)
        rows = torch.arange(len(state_indices))
        states[rows, torch.tensor(state_indices, dtype=torch.long)] = 1.0
        return states.to(self.device)

    def select_action(self, state: int) -> int:
        # Explore with probability epsilon, otherwise exploit learned Q-values.
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = self._one_hot(np.array([state]))
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        # Experience tuple for off-policy learning.
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def train_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            # Warm-up phase: wait until replay buffer has enough samples.
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states = np.array([t.state for t in batch], dtype=np.int64)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = np.array([t.next_state for t in batch], dtype=np.int64)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)

        state_tensor = self._one_hot(states)
        next_state_tensor = self._one_hot(next_states)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a) estimates for actions that were actually taken.
        q_values = self.online_net(state_tensor).gather(1, actions)
        with torch.no_grad():
            # Target network stabilizes training targets.
            next_q_max = self.target_net(next_state_tensor).max(dim=1, keepdim=True)[0]
            # Bellman backup with terminal masking.
            target_q = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Single optimizer step on online network parameters.
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            # Periodic hard update from online -> target network.
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        # Gradually reduce exploration as the policy improves.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def run_dqn_agent(
    env: CyberKillChainEnv, episodes: int
) -> Tuple[List[float], List[int], List[float], Dict[str, float]]:
    """Train DQN agent and collect rewards, steps, and average losses."""
    agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions)
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_losses: List[float] = []
    wrong_actions_total = 0
    failed_episodes = 0

    # Standard episodic training loop.
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        losses: List[float] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if bool(info.get("wrong_action", False)):
                wrong_actions_total += 1

            # Store transition then update online network from replayed mini-batches.
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0.0:
                losses.append(loss)

            total_reward += reward
            steps += 1
            state = next_state

        # Epsilon schedule updates once per episode.
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        # Mean per-episode loss is useful for debugging training stability.
        episode_losses.append(float(np.mean(losses)) if losses else 0.0)
        if state != env.terminal_state:
            failed_episodes += 1

    stats = {
        # Same schema as random baseline stats to simplify reporting logic.
        "wrong_actions_total": float(wrong_actions_total),
        "wrong_actions_per_episode": float(wrong_actions_total) / float(episodes),
        "failed_episodes": float(failed_episodes),
        "failure_rate": float(failed_episodes) / float(episodes),
    }
    return episode_rewards, episode_steps, episode_losses, stats
