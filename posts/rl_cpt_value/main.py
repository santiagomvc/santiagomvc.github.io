"""CliffWalking environment runner with CLI."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agents import AGENTS, get_agent
from custom_cliff_walking import make_env
from utils import load_config, save_episodes_gif, save_training_curves


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
        fell_off_cliff = reward == cfg["reward_cliff"]
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


def evaluate(env, agent, agent_name, config_name="base"):
    """Run N episodes, save GIFs, and print metrics summary."""
    cfg = load_config(config_name)
    n_episodes = cfg["n_eval_episodes"]
    all_metrics = []
    all_frames = []

    for _ in range(n_episodes):
        frames, metrics = run_episode(env, agent, config_name=config_name)
        all_frames.append(frames)
        all_metrics.append(metrics)

    output_path = f"outputs/{agent_name}_{ENV_NAME}.gif"
    save_episodes_gif(all_frames, output_path)
    print(f"Saved {output_path}")

    # Print summary
    avg_reward = sum(m["total_reward"] for m in all_metrics) / n_episodes
    avg_length = sum(m["episode_length"] for m in all_metrics) / n_episodes
    success_rate = sum(m["success"] for m in all_metrics) / n_episodes
    total_cliff_falls = sum(m["cliff_falls"] for m in all_metrics)
    print(f"\nAgent: {agent_name}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Length: {avg_length:.2f}")
    print(f"  Success Rate: {success_rate:.0%}")
    print(f"  Total Cliff Falls: {total_cliff_falls}")


def parse_args():
    parser = argparse.ArgumentParser(description="CliffWalking RL runner")
    parser.add_argument("-a", "--agent", choices=list(AGENTS.keys()), required=True)
    parser.add_argument("-c", "--config", default="base", help="Config name from configs/ folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    cfg = load_config(args.config)
    env = make_env(args.config)
    print(f"Using CliffWalking (shape={tuple(cfg['shape'])}, stochasticity={cfg['stochasticity']})")

    agent = get_agent(args.agent, env)

    if agent.trainable:
        print(f"Training {args.agent} for {cfg['timesteps']} timesteps...")
        history = agent.learn(env, cfg["timesteps"])
        save_training_curves(history, "outputs", args.agent)
        print("Training complete. Running evaluation...")

    evaluate(env, agent, args.agent, config_name=args.config)

    agent.close()
    env.close()
