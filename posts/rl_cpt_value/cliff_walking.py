"""CliffWalking environment runner."""

import gymnasium as gym
from agents import random_agent
from utils import save_gif, GOAL


def run_episode(env, agent_fn, max_steps=500):
    """Run one episode, return states and rewards."""
    state, _ = env.reset()
    states, rewards = [state], []

    for _ in range(max_steps):
        action = agent_fn(env)
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        if terminated or truncated:
            break

    return states, rewards


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1")

    print("Running 5 episodes with random agent...")
    print("-" * 40)

    for ep in range(5):
        states, rewards = run_episode(env, random_agent)
        total = sum(rewards)
        status = "GOAL" if states[-1] == GOAL else "CLIFF/TIMEOUT"
        print(f"Episode {ep+1}: {status} | Steps: {len(rewards)} | Reward: {total:.0f}")

        if ep == 0:
            save_gif(states, rewards, "outputs/random_agent.gif")
            print("  -> Saved GIF")

    env.close()
