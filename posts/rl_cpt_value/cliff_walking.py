"""CliffWalking environment runner with CLI."""

import argparse
import gymnasium as gym
from agents import get_agent, AGENTS
from utils import save_gif

ENV_NAME = "CliffWalking-v0"
TIMESTEPS = 5000
N_EVAL_EPISODES = 5


def run_episode(env, agent, max_steps=500):
    """Run one episode. Works for any agent with act(state)."""
    state, _ = env.reset()
    states, rewards, frames = [state], [], [env.render()]

    for _ in range(max_steps):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        frames.append(env.render())
        if terminated or truncated:
            break

    return states, rewards, frames


def evaluate(env, agent, agent_name):
    """Run N episodes and save GIFs."""
    for i in range(N_EVAL_EPISODES):
        states, rewards, frames = run_episode(env, agent)
        output = f"outputs/{agent_name}_{ENV_NAME}_ep{i+1}.gif"
        save_gif(frames, output)
        print(f"Episode {i+1}: reward={sum(rewards)}, saved={output}")


def parse_args():
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-t", "--type", choices=["train", "eval"], required=True)
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = gym.make(ENV_NAME, render_mode="rgb_array")
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
