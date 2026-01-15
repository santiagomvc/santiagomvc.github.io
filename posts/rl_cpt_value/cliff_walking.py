"""CliffWalking environment runner with CLI."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
from dotenv import load_dotenv

from agents import AGENTS, get_agent
from utils import save_gif


ENV_NAME = "CliffWalking-v0"
TIMESTEPS = 75000
N_EVAL_EPISODES = 20


def run_episode(env, agent, max_steps=500):
    """Run one episode. Works for any agent with act(state).

    Returns: states, rewards, frames, stats dict
    """
    state, _ = env.reset()
    states, rewards, frames = [state], [], [env.render()]
    cliff_falls = 0
    safe_visits = 0

    for _ in range(max_steps):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        frames.append(env.render())

        # Track stats
        if reward == -100:
            cliff_falls += 1
        row = state // 12
        if row < 3:
            safe_visits += 1

        if terminated or truncated:
            break

    stats = {
        "reward": sum(rewards),
        "steps": len(rewards),
        "cliff_falls": cliff_falls,
        "safe_row_pct": round(safe_visits / len(rewards) * 100, 1) if rewards else 0,
        "reached_goal": state == 47,
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
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-t", "--type", choices=["train", "eval"], required=True)
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load environment variables
    load_dotenv(Path(__file__).parent / ".env")

    env = gym.make(ENV_NAME, render_mode="rgb_array", is_slippery=True)
    agent = get_agent(args.agent, env)

    if args.type == "train":
        print(f"Training {args.agent} for {TIMESTEPS} timesteps...")
        agent.learn(env, TIMESTEPS)
        print("Training complete. Running evaluation...")
        evaluate(env, agent, args.agent)
    else:
        evaluate(env, agent, args.agent)

    agent.close()
    env.close()
