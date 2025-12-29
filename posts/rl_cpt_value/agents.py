"""Agent classes."""

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig


class RandomAgent:
    """Random action agent."""

    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def act(self, state):
        return np.random.randint(self.n_actions)


class PPOAgent:
    """PPO agent using RLlib."""

    def __init__(self, env_name="CliffWalking-v0"):
        ray.init(ignore_reinit_error=True)
        config = PPOConfig().environment(env_name).env_runners(num_env_runners=0)
        self.algo = config.build()

    def act(self, state):
        return self.algo.compute_single_action(state)

    def learn(self, iterations=50):
        for i in range(iterations):
            result = self.algo.train()
            reward = result.get("env_runners", {}).get("episode_reward_mean", "N/A")
            print(f"Iter {i+1}: reward={reward}")

    def close(self):
        self.algo.stop()
        ray.shutdown()
