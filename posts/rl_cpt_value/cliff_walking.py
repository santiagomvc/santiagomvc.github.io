"""CliffWalking environment runner."""

import gymnasium as gym
from agents import RandomAgent, PPOAgent
from utils import save_gif


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


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")

    # Random agent (no training)
    print("Random Agent:")
    agent = RandomAgent(n_actions=env.action_space.n)
    states, rewards, frames = run_episode(env, agent)
    save_gif(frames, "outputs/random_agent.gif")
    print(f"  Reward: {sum(rewards)}")

    # PPO agent (train then evaluate)
    print("\nPPO Agent:")
    agent = PPOAgent()
    agent.learn(iterations=50)
    states, rewards, frames = run_episode(env, agent)
    save_gif(frames, "outputs/ppo_agent.gif")
    print(f"  Reward: {sum(rewards)}")
    agent.close()

    env.close()
