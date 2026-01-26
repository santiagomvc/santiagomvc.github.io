"""Configurable CliffWalking environment and wrappers."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv, UP, RIGHT, DOWN, LEFT

from utils import load_config


class ResizableCliffWalkingEnv(CliffWalkingEnv):
    """CliffWalking with configurable grid dimensions."""

    def __init__(self, render_mode=None, is_slippery=False, shape=(5, 5)):
        super().__init__(render_mode=render_mode, is_slippery=is_slippery)

        # Override shape-dependent attributes
        self.shape = shape
        nrows, ncols = shape
        self.start_state_index = np.ravel_multi_index((nrows - 1, 0), self.shape)
        self.nS = np.prod(self.shape)

        # Rebuild cliff (bottom row between start and goal)
        self._cliff = np.zeros(self.shape, dtype=bool)
        if ncols > 2:
            self._cliff[nrows - 1, 1:-1] = True

        # Rebuild transition probabilities
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            for action in [UP, RIGHT, DOWN, LEFT]:
                self.P[s][action] = self._calculate_transition_prob(position, action)

        # Rebuild initial state distribution
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        # Update observation space
        self.observation_space = spaces.Discrete(self.nS)

        # Reset rendering for custom shape
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None


class CliffWalkingWrapper(gym.Wrapper):
    """Combined wrapper for rewards, termination, and wind."""

    def __init__(
        self,
        env,
        reward_cliff=-100.0,
        reward_step=-1.0,
        terminate_on_cliff=True,
        wind_prob=0.0,
    ):
        super().__init__(env)
        self.reward_cliff = reward_cliff
        self.reward_step = reward_step
        self.terminate_on_cliff = terminate_on_cliff
        self.wind_prob = wind_prob

    def step(self, action):
        # Wind: maybe change action to DOWN
        if self.wind_prob > 0 and self.np_random.random() < self.wind_prob:
            action = 2  # DOWN

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Terminate on cliff (check before reward transform)
        if self.terminate_on_cliff and reward == -100:
            terminated = True

        # Transform rewards
        if reward == -100:
            reward = self.reward_cliff
        elif reward == -1:
            reward = self.reward_step

        return obs, reward, terminated, truncated, info


def make_env(config_name: str = "base"):
    """Factory function to create CliffWalking environment from config."""
    cfg = load_config(config_name)
    shape = tuple(cfg["shape"])
    stochasticity = cfg["stochasticity"]
    env = ResizableCliffWalkingEnv(
        render_mode="rgb_array",
        is_slippery=(stochasticity == "slippery"),
        shape=shape,
    )
    env = CliffWalkingWrapper(
        env,
        reward_cliff=cfg["reward_cliff"],
        reward_step=cfg["reward_step"],
        terminate_on_cliff=True,
        wind_prob=cfg["wind_prob"] if stochasticity == "windy" else 0.0,
    )
    return env
