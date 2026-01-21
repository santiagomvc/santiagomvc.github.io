"""CliffWalking environment runner with CLI."""

import argparse
from pathlib import Path

import gymnasium as gym
from dotenv import load_dotenv

from agents import AGENTS, get_agent
from utils import save_gif


TIMESTEPS = 300000
N_EVAL_EPISODES = 5
ENV_NAME = "CliffWalking-v0"

# Slippery mode: adds stochastic transitions like FrozenLake's is_slippery
USE_SLIPPERY = True


def run_episode(env, agent, max_steps=500):
    """Run one episode and collect frames for GIF."""
    state, _ = env.reset()
    frames = [env.render()]

    for _ in range(max_steps):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            break

    return frames


def evaluate(env, agent, agent_name):
    """Run N episodes and save GIFs."""
    for i in range(N_EVAL_EPISODES):
        frames = run_episode(env, agent)
        output = f"outputs/{agent_name}_{ENV_NAME}_ep{i+1}.gif"
        save_gif(frames, output)
        print(f"Episode {i+1}: saved {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-t", "--type", choices=["train", "eval"], required=True)
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    env = gym.make("CliffWalking-v0", render_mode="rgb_array", is_slippery=USE_SLIPPERY)
    print(f"Using CliffWalking-v0 (slippery={USE_SLIPPERY})")

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
