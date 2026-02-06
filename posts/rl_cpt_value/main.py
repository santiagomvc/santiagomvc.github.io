"""CliffWalking environment runner with CLI."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from agents import get_agent
from custom_cliff_walking import make_env
from utils import load_config, save_episodes_gif, save_training_curves, evaluate_paths, summarize_paths


ENV_NAME = "CliffWalking-v1"


def run_episode(env, agent, max_steps=500, config_name="base"):
    """Run one episode and collect frames for GIF."""
    state, _ = env.reset()
    frames = [env.render()]
    total_reward = 0
    step_count = 0
    cliff_falls = 0

    cfg = load_config(config_name)
    fell_off_cliff = False
    for _ in range(max_steps):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward
        step_count += 1
        fell_off_cliff = reward == cfg["env"]["reward_cliff"]
        if fell_off_cliff:
            cliff_falls += 1

        if terminated or truncated:
            break

    metrics = {
        "total_reward": total_reward,
        "episode_length": step_count,
        "success": terminated and not fell_off_cliff,
        "cliff_falls": cliff_falls,
    }
    return frames, metrics


def evaluate(env, agent, output_dir, config_name="base"):
    """Run N episodes, save GIFs, and print metrics summary."""
    cfg = load_config(config_name)
    n_episodes = cfg["training"]["n_eval_episodes"]
    all_metrics = []
    all_frames = []

    for _ in range(n_episodes):
        frames, metrics = run_episode(env, agent, config_name=config_name)
        all_frames.append(frames)
        all_metrics.append(metrics)

    output_path = output_dir / "eval.gif"
    save_episodes_gif(all_frames, str(output_path))
    print(f"Saved {output_path}")

    # Print summary
    avg_reward = sum(m["total_reward"] for m in all_metrics) / n_episodes
    avg_length = sum(m["episode_length"] for m in all_metrics) / n_episodes
    success_rate = sum(m["success"] for m in all_metrics) / n_episodes
    total_cliff_falls = sum(m["cliff_falls"] for m in all_metrics)
    print(f"\nAgent: {output_dir.name}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Length: {avg_length:.2f}")
    print(f"  Success Rate: {success_rate:.0%}")
    print(f"  Total Cliff Falls: {total_cliff_falls}")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-c", "--config", nargs="+", default=["base"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    for config_name in args.config:
        cfg = load_config(config_name)

        for agent_entry in cfg["agents"]:
            if isinstance(agent_entry, str):
                agent_name = agent_entry
                agent_overrides = {}
            elif "name" in agent_entry:
                # Format: {"name": "agent-name", "key": "value", ...}
                agent_name = agent_entry["name"]
                agent_overrides = {k: v for k, v in agent_entry.items() if k != "name"}
            else:
                # Format: {"agent-name": {"key": "value", ...}}
                agent_name = list(agent_entry.keys())[0]
                agent_overrides = agent_entry[agent_name] or {}

            # Merge: defaults + per-agent overrides
            agent_cfg = {**cfg["agent_config"], **agent_overrides}

            n_seeds = cfg["training"]["n_seeds"]
            nrows, ncols = cfg["env"]["shape"]
            n_eval = cfg["training"]["n_eval_episodes"]
            all_path_results = []

            for seed in range(1, n_seeds + 1):
                suffix = f"_seed{seed}" if n_seeds > 1 else ""
                output_dir = Path(f"outputs/{agent_name}_{config_name}{suffix}")
                output_dir.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                env = make_env(config_name, seed=seed)

                if agent_name == "cpt-reinforce":
                    agent_cfg["env_config"] = cfg
                agent = get_agent(agent_name, env, **agent_cfg)

                if agent.trainable:
                    print(f"Training {agent_name} for {cfg['training']['timesteps']} timesteps...")
                    history = agent.learn(
                        env, cfg["training"]["timesteps"],
                        batch_size=cfg["training"]["batch_size"],
                        entropy_coef=cfg["training"]["entropy_coef"],
                        entropy_coef_final=cfg["training"]["entropy_coef_final"],
                    )
                    save_training_curves(history, str(output_dir), agent_name)
                    np.savez(
                        output_dir / "history.npz",
                        episode_rewards=np.array(history["episode_rewards"]),
                        batch_losses=np.array(history["batch_losses"]),
                    )

                evaluate(env, agent, output_dir, config_name=config_name)
                path_result = evaluate_paths(env, agent, n_eval, config_name=config_name)
                all_path_results.append(path_result)
                agent.close()
                env.close()

            summarize_paths(all_path_results, nrows, agent_name, config_name)
