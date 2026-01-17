"""FrozenLake environment runner with CLI."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv

from agents import AGENTS, get_agent
from utils import compute_action_entropy, save_gif


ENV_NAME = "FrozenLake-v1"
GRID_SIZE = 4
HOLES = {5, 7, 11, 12}
GOAL = 15
# Tiles adjacent to holes (risky to visit)
RISKY_TILES = {1, 3, 4, 6, 8, 9, 10, 13, 15}
TIMESTEPS = 150000
N_EVAL_EPISODES = 20
OPTIMAL_STEPS = 6  # Minimum steps from start to goal


def run_episode(env, agent, max_steps=200):
    """Run one episode. Works for any agent with act(state).

    Returns: states, rewards, frames, stats dict
    """
    state, _ = env.reset()
    states, rewards, frames = [state], [], [env.render()]
    actions = []
    safe_visits = 0

    for _ in range(max_steps):
        action = agent.act(state)
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        frames.append(env.render())

        # Track safe zone visits (not adjacent to holes)
        if state not in RISKY_TILES and state not in HOLES:
            safe_visits += 1

        if terminated or truncated:
            break

    # Compute behavioral metrics
    action_counts = np.bincount(actions, minlength=4) if actions else np.zeros(4)
    action_dist = (action_counts / len(actions) * 100).tolist() if actions else [0, 0, 0, 0]
    unique_states = len(set(states))

    stats = {
        "reward": sum(rewards),
        "steps": len(rewards),
        "fell_in_hole": state in HOLES,
        "safe_zone_pct": round(safe_visits / len(rewards) * 100, 1) if rewards else 0,
        "reached_goal": state == GOAL,
        "action_entropy": compute_action_entropy(actions),
        "path_efficiency": round(OPTIMAL_STEPS / len(rewards), 3) if rewards else 0,
        "unique_states": unique_states,
        "action_dist": [round(p, 1) for p in action_dist],  # [left, down, right, up]
    }
    return states, rewards, frames, stats


def evaluate(env, agent, agent_name):
    """Run N episodes, save GIFs and stats JSON."""
    all_stats = []
    for i in range(N_EVAL_EPISODES):
        states, rewards, frames, stats = run_episode(env, agent)
        all_stats.append(stats)
        output = f"outputs/{agent_name}_{ENV_NAME}_ep{i+1}.gif"
        save_gif(frames, output)
        print(f"Episode {i+1}: reward={stats['reward']}, saved={output}")

    # Save stats JSON
    stats_path = Path(f"outputs/{agent_name}_{ENV_NAME}_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"agent": agent_name, "env": ENV_NAME, "episodes": all_stats}, f, indent=2)
    print(f"Stats saved to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="FrozenLake RL runner")
    parser.add_argument("-t", "--type", choices=["train", "eval"], required=True)
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load environment variables
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    env = gym.make(ENV_NAME, render_mode="rgb_array", is_slippery=True)
    agent = get_agent(args.agent, env, env_name=ENV_NAME)

    if args.type == "train":
        print(f"Training {args.agent} for {TIMESTEPS} timesteps...")
        agent.learn(env, TIMESTEPS)
        print("Training complete. Running evaluation...")
        evaluate(env, agent, args.agent)
    else:
        evaluate(env, agent, args.agent)

    agent.close()
    env.close()
