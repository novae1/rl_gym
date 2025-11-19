from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Feed-forward network with Tanh activations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SeededGymEnv:
    """Wrapper that exposes deterministic resets compatible with GRPO."""

    def __init__(self, env_id: str, **make_kwargs) -> None:
        self.env_id = env_id
        self.make_kwargs = make_kwargs
        self.env = gym.make(env_id, **make_kwargs)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class LivePlotter:
    """Matplotlib helper for lightweight live-updating curves."""

    def __init__(self, title: str = "Training", ylabel: str = "Return") -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Update")
        self.ax.set_ylabel(ylabel)
        (self.line,) = self.ax.plot([], [], label="avg return")
        self.ax.legend()
        self.x_data: List[int] = []
        self.y_data: List[float] = []

    def update(self, step: int, value: float) -> None:
        self.x_data.append(step)
        self.y_data.append(value)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


class PPOBuffer:
    """Stores on-policy transitions and computes GAE advantages."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_buf = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device(device)
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        assert self.ptr < self.max_size, "PPOBuffer overflow"
        self.obs_buf[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        if self.path_start_idx == self.ptr:
            return
        last_advantage = 0.0
        last_return = last_value
        end = self.ptr - 1
        for idx in range(end, self.path_start_idx - 1, -1):
            nonterminal = 1.0 - self.dones[idx]
            next_value = last_value if idx == end else self.values[idx + 1]
            delta = self.rewards[idx] + self.gamma * next_value * nonterminal - self.values[idx]
            last_advantage = delta + self.gamma * self.lam * nonterminal * last_advantage
            self.advantages[idx] = last_advantage
            last_return = self.values[idx] + last_advantage
            self.returns[idx] = last_return
        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.ptr == self.max_size, "Buffer not full"
        adv = self.advantages[: self.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(
            obs=torch.as_tensor(self.obs_buf[: self.ptr], device=self.device),
            actions=torch.as_tensor(self.actions[: self.ptr], device=self.device),
            log_probs=torch.as_tensor(self.log_probs[: self.ptr], device=self.device),
            advantages=torch.as_tensor(adv, device=self.device),
            returns=torch.as_tensor(self.returns[: self.ptr], device=self.device),
        )
        return data

    def reset(self) -> None:
        self.ptr = 0
        self.path_start_idx = 0


class GRPOBuffer:
    """Episode-oriented buffer for GRPO outcome supervision."""

    def __init__(self, capacity: int, device: str | torch.device = "cpu") -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.episodes: List[Dict[str, torch.Tensor]] = []
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.current_episode: Dict[str, List[float]] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
        }

    def store(self, state, action: int, reward: float, log_prob: float) -> None:
        flat_state = np.asarray(state, dtype=np.float32).reshape(-1)
        self.current_episode["states"].append(flat_state)
        self.current_episode["actions"].append(int(action))
        self.current_episode["rewards"].append(float(reward))
        self.current_episode["log_probs"].append(float(log_prob))

    def end_episode(self) -> Optional[float]:
        rewards = self.current_episode["rewards"]
        if not rewards:
            return None
        states = torch.as_tensor(
            np.asarray(self.current_episode["states"], dtype=np.float32),
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(self.current_episode["actions"], dtype=np.int64),
            device=self.device,
        )
        log_probs = torch.as_tensor(
            np.asarray(self.current_episode["log_probs"], dtype=np.float32),
            device=self.device,
        )
        reward_array = np.asarray(rewards, dtype=np.float32)
        total_return = float(reward_array.sum())
        length = len(reward_array)
        weight = 1.0 / max(length, 1)

        self.episodes.append(
            {
                "states": states,
                "actions": actions,
                "log_probs": log_probs,
                "return": torch.tensor(total_return, device=self.device),
                "weight": torch.tensor(weight, device=self.device),
                "length": length,
            }
        )
        self._reset_episode()
        return total_return

    def is_full(self) -> bool:
        return len(self.episodes) >= self.capacity

    def get(self) -> List[Dict[str, torch.Tensor]]:
        return list(self.episodes)

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
