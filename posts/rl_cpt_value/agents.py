"""Agent classes with Stable Baselines 3."""

from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3 import PPO


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def act(self, state):
        """Select action given state."""
        pass

    def learn(self, env, timesteps):
        """Train the agent. Override if trainable."""
        pass

    def close(self):
        """Cleanup resources. Override if needed."""
        pass


class RandomAgent(BaseAgent):
    """Random action agent."""

    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def act(self, state):
        return np.random.randint(self.n_actions)


class PPOAgent(BaseAgent):
    """PPO agent using Stable Baselines 3."""

    def __init__(self, env):
        """Initialize with environment instance."""
        self.model = PPO("MlpPolicy", env, verbose=1)

    def act(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def learn(self, env, timesteps=50000):
        """Train for given timesteps."""
        self.model.learn(total_timesteps=timesteps)


# Agent registry for extensibility
AGENTS = {
    "random": RandomAgent,
    "ppo": PPOAgent,
}


def get_agent(name, env):
    """Factory function to create agent by name."""
    if name not in AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENTS.keys())}")

    if name == "random":
        return AGENTS[name](n_actions=env.action_space.n)
    return AGENTS[name](env)
