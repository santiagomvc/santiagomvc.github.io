"""CliffWalking environment runner with CLI."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

import config
from agents import AGENTS, get_agent
from custom_cliff_walking import make_env
from utils import save_gif


ENV_NAME = "CliffWalking-v1"


def run_episode(env, agent, max_steps=500):
    """Run one episode and collect frames for GIF."""
    state, _ = env.reset()
    frames = [env.render()]
    total_reward = 0
    step_count = 0
    cliff_falls = 0

    fell_off_cliff = False
    for _ in range(max_steps):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward
        step_count += 1
        fell_off_cliff = reward == config.REWARD_CLIFF
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


def evaluate(env, agent, agent_name):
    """Run N episodes, save GIFs, and print metrics summary."""
    all_metrics = []

    for i in range(config.N_EVAL_EPISODES):
        frames, metrics = run_episode(env, agent)
        all_metrics.append(metrics)
        output = f"outputs/{agent_name}_{ENV_NAME}_ep{i+1}.gif"
        save_gif(frames, output)
        print(f"Episode {i+1}: saved {output}")

    # Print summary
    avg_reward = sum(m["total_reward"] for m in all_metrics) / config.N_EVAL_EPISODES
    avg_length = sum(m["episode_length"] for m in all_metrics) / config.N_EVAL_EPISODES
    success_rate = sum(m["success"] for m in all_metrics) / config.N_EVAL_EPISODES
    total_cliff_falls = sum(m["cliff_falls"] for m in all_metrics)
    print(f"\nAgent: {agent_name}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Length: {avg_length:.2f}")
    print(f"  Success Rate: {success_rate:.0%}")
    print(f"  Total Cliff Falls: {total_cliff_falls}")


def parse_args():
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-t", "--type", choices=["train", "eval"], required=True)
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    env = make_env(config)
    print(f"Using CliffWalking (shape={config.SHAPE}, stochasticity={config.STOCHASTICITY})")

    agent = get_agent(args.agent, env)

    if args.type == "train":
        print(f"Training {args.agent} for {config.TIMESTEPS} timesteps...")
        agent.learn(env, config.TIMESTEPS)
        print("Training complete. Running evaluation...")
        evaluate(env, agent, args.agent)
    else:
        evaluate(env, agent, args.agent)

    agent.close()
    env.close()
