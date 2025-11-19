from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from algorithms import GRPOAgent, PPOAgent
from utils import GRPOBuffer, LivePlotter, PPOBuffer, SeededGymEnv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG: Dict = {
    "env_id": "CartPole-v1",
    "seed": 42,
    "device": DEFAULT_DEVICE,
    "checkpoint_dir": BASE_DIR / "checkpoints",
    "ppo": {
        "steps_per_update": 2048,
        "total_updates": 150,
        "hidden_sizes": (64, 64),
        "lr": 3e-4,
        "clip_range": 0.2,
        "update_epochs": 10,
        "minibatch_size": 64,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "save_interval": 10,
        "rolling_window": 10,
    },
    "grpo": {
        "group_size": 16,
        "episodes_per_update": 32,
        "total_updates": 150,
        "hidden_sizes": (64, 64),
        "lr": 3e-4,
        "clip_range": 0.2,
        "save_interval": 10,
        "rolling_window": 10,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_ppo(config: Dict) -> None:
    ppo_cfg = config["ppo"]
    env = SeededGymEnv(config["env_id"])
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    device = config["device"]
    agent_cfg = {**ppo_cfg, "device": device}
    agent = PPOAgent(obs_dim, act_dim, agent_cfg)
    buffer = PPOBuffer(
        obs_shape=(obs_dim,),
        size=ppo_cfg["steps_per_update"],
        gamma=ppo_cfg["gamma"],
        lam=ppo_cfg["gae_lambda"],
        device=device,
    )

    obs_raw, _ = env.reset(seed=config["seed"])
    obs = flatten_obs(obs_raw)
    episode_return = 0.0
    episode_returns = []
    best_avg = float("-inf")
    ckpt_dir = Path(config["checkpoint_dir"]) / "ppo"
    ensure_dir(ckpt_dir)
    plotter = LivePlotter("PPO Training", "Average Return")

    for update in range(1, ppo_cfg["total_updates"] + 1):
        for step in range(ppo_cfg["steps_per_update"]):
            action, log_prob, value = agent.get_action(obs)
            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(obs, action, reward, value, log_prob, done)
            episode_return += reward
            next_obs = flatten_obs(next_obs_raw)
            obs = next_obs

            if done:
                buffer.finish_path(last_value=0.0)
                episode_returns.append(episode_return)
                obs_raw, _ = env.reset()
                obs = flatten_obs(obs_raw)
                episode_return = 0.0

        buffer.finish_path(last_value=agent.value(obs))
        stats = agent.update(buffer.get())
        buffer.reset()

        avg_return = np.mean(episode_returns[-ppo_cfg["rolling_window"] :]) if episode_returns else 0.0
        plotter.update(update, avg_return)
        print(
            f"[PPO] Update {update:04d} | Avg Return {avg_return:.2f} | "
            f"Policy Loss {stats['policy_loss']:.4f} | Value Loss {stats['value_loss']:.4f}"
        )

        if update % ppo_cfg["save_interval"] == 0:
            save_path = ckpt_dir / f"ppo_update_{update:04d}.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "config": agent_cfg,
                    "update": update,
                },
                save_path,
            )

        if avg_return > best_avg and episode_returns:
            best_avg = avg_return
            best_path = ckpt_dir / "ppo_best.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "config": agent_cfg,
                    "update": update,
                    "avg_return": avg_return,
                },
                best_path,
            )

    plotter.close()
    env.close()


def train_grpo(config: Dict) -> None:
    grpo_cfg = config["grpo"]
    if grpo_cfg["episodes_per_update"] % grpo_cfg["group_size"] != 0:
        raise ValueError("episodes_per_update must be divisible by group_size.")
    env = SeededGymEnv(config["env_id"])
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    device = config["device"]
    agent_cfg = {**grpo_cfg, "device": device}
    agent = GRPOAgent(obs_dim, act_dim, agent_cfg)
    buffer = GRPOBuffer(capacity=grpo_cfg["episodes_per_update"], device=device)
    episode_returns = []
    best_avg = float("-inf")
    ckpt_dir = Path(config["checkpoint_dir"]) / "grpo"
    ensure_dir(ckpt_dir)
    plotter = LivePlotter("GRPO Training", "Average Return")

    for update in range(1, grpo_cfg["total_updates"] + 1):
        buffer.clear()
        while len(buffer) < grpo_cfg["episodes_per_update"]:
            seed = random.randint(0, 2**31 - 1)
            for _ in range(grpo_cfg["group_size"]):
                obs_raw, _ = env.reset(seed=seed)
                obs = flatten_obs(obs_raw)
                done = False
                while not done:
                    action, log_prob = agent.get_action(obs)
                    next_obs_raw, reward, terminated, truncated, _ = env.step(action)
                    buffer.store(obs, action, reward, log_prob)
                    obs = flatten_obs(next_obs_raw)
                    done = terminated or truncated
                ret = buffer.end_episode()
                if ret is not None:
                    episode_returns.append(ret)
                if buffer.is_full():
                    break

        stats = agent.update(buffer.get(), grpo_cfg["group_size"])
        buffer.clear()
        avg_return = np.mean(episode_returns[-grpo_cfg["rolling_window"] :]) if episode_returns else 0.0
        plotter.update(update, avg_return)
        print(f"[GRPO] Update {update:04d} | Avg Return {avg_return:.2f} | Loss {stats['loss']:.4f}")

        if update % grpo_cfg["save_interval"] == 0:
            save_path = ckpt_dir / f"grpo_update_{update:04d}.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "config": agent_cfg,
                    "update": update,
                },
                save_path,
            )

        if avg_return > best_avg and episode_returns:
            best_avg = avg_return
            best_path = ckpt_dir / "grpo_best.pt"
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "config": agent_cfg,
                    "update": update,
                    "avg_return": avg_return,
                },
                best_path,
            )

    plotter.close()
    env.close()


def play(config: Dict, agent_type: str, checkpoint: Path, episodes: int, render: bool) -> None:
    env_kwargs = {"render_mode": "human"} if render else {}
    env = SeededGymEnv(config["env_id"], **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    device = config["device"]

    if agent_type == "ppo":
        agent = PPOAgent(obs_dim, act_dim, {**config["ppo"], "device": device})
        state = torch.load(checkpoint, map_location=device)
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
    elif agent_type == "grpo":
        agent = GRPOAgent(obs_dim, act_dim, {**config["grpo"], "device": device})
        state = torch.load(checkpoint, map_location=device)
        agent.actor.load_state_dict(state["actor"])
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent.actor.eval()
    if isinstance(agent, PPOAgent):
        agent.critic.eval()

    for ep in range(episodes):
        obs_raw, _ = env.reset(seed=config["seed"])
        obs = flatten_obs(obs_raw)
        done = False
        total_reward = 0.0
        while not done:
            if agent_type == "ppo":
                action, _, _ = agent.get_action(obs)
            else:
                action, _ = agent.get_action(obs)
            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            obs = flatten_obs(next_obs_raw)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
        print(f"Play Episode {ep + 1}: return={total_reward:.2f}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare PPO vs GRPO on CartPole-v1")
    parser.add_argument("--mode", choices=["ppo", "grpo", "play"], default="ppo")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for play mode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of play episodes")
    parser.add_argument("--agent", choices=["ppo", "grpo"], help="Agent type for play mode")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])

    if args.mode == "ppo":
        train_ppo(CONFIG)
    elif args.mode == "grpo":
        train_grpo(CONFIG)
    else:
        if not args.checkpoint or not args.agent:
            raise ValueError("Play mode requires --checkpoint and --agent arguments.")
        play(CONFIG, args.agent, args.checkpoint, args.episodes, render=True)


if __name__ == "__main__":
    main()
