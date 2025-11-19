from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.distributions import Categorical

from utils import MLP


class PPOAgent:
    """Standard clipped PPO actor-critic agent."""

    def __init__(self, obs_dim: int, act_dim: int, config: Dict) -> None:
        self.device = torch.device(config.get("device", "cpu"))
        hidden_sizes = tuple(config.get("hidden_sizes", (64, 64)))
        self.actor = MLP(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic = MLP(obs_dim, 1, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.get("lr", 3e-4),
        )
        self.clip_range = config.get("clip_range", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.value_coef = config.get("value_coef", 0.5)
        self.update_epochs = config.get("update_epochs", 10)
        self.minibatch_size = config.get("minibatch_size", 64)

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_tensor).squeeze(-1)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def value(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        return float(self.critic(obs_tensor).squeeze(-1).item())

    def _evaluate(self, obs_batch: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs_batch).squeeze(-1)
        return log_probs, entropy, values

    def update(self, buffer_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = buffer_data["obs"]
        actions = buffer_data["actions"]
        old_log_probs = buffer_data["log_probs"]
        advantages = buffer_data["advantages"]
        returns = buffer_data["returns"]
        batch_size = obs.shape[0]
        clip = self.clip_range

        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        updates = 0

        for _ in range(self.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs_old = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                log_probs, entropy, values = self._evaluate(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_log_probs_old)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                policy_loss = -torch.mean(torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages))
                value_loss = torch.mean((values - mb_returns) ** 2)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    0.5,
                )
                self.optimizer.step()

                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                entropy_accum += entropy.mean().item()
                updates += 1

        return {
            "policy_loss": policy_loss_accum / max(updates, 1),
            "value_loss": value_loss_accum / max(updates, 1),
            "entropy": entropy_accum / max(updates, 1),
        }


class GRPOAgent:
    """Actor-only agent trained via Group Relative Policy Optimization."""

    def __init__(self, obs_dim: int, act_dim: int, config: Dict) -> None:
        self.device = torch.device(config.get("device", "cpu"))
        hidden_sizes = tuple(config.get("hidden_sizes", (64, 64)))
        self.actor = MLP(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("lr", 3e-4))
        self.clip_range = config.get("clip_range", 0.2)
        self.adv_eps = config.get("adv_eps", 1e-8)

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def update(self, episodes: List[Dict[str, torch.Tensor]], group_size: int) -> Dict[str, float]:
        if not episodes:
            return {"loss": 0.0}
        clip = self.clip_range
        advantages: List[torch.Tensor] = []
        device = self.device

        for start in range(0, len(episodes), group_size):
            chunk = episodes[start : start + group_size]
            chunk_returns = torch.stack([ep["return"] for ep in chunk]).to(device)
            mean = torch.mean(chunk_returns)
            std = torch.std(chunk_returns, unbiased=False)
            std = torch.clamp(std, min=1e-6)
            chunk_advantages = (chunk_returns - mean) / (std + self.adv_eps)
            for adv in chunk_advantages:
                advantages.append(adv)

        total_loss = torch.tensor(0.0, device=device)
        total_steps = 0
        self.optimizer.zero_grad()

        for episode, adv in zip(episodes, advantages):
            states = episode["states"].to(device)
            actions = episode["actions"].to(device)
            old_log_probs = episode["log_probs"].to(device)
            weight = episode["weight"].to(device)
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            loss_terms = torch.min(ratio * adv, clipped_ratio * adv)
            episode_loss = -weight * torch.sum(loss_terms)
            total_loss = total_loss + episode_loss
            total_steps += episode["length"]

        total_steps = max(total_steps, 1)
        mean_loss = total_loss / total_steps
        mean_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer.step()

        return {"loss": float(mean_loss.item())}
