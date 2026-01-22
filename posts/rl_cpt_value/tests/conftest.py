"""Pytest fixtures for CPT CliffWalking diagnostics."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_cliff_walking import ResizableCliffWalkingEnv, CliffWalkingWrapper
from utils import CPTValueFunction
from agents import REINFORCEAgent, CPTREINFORCEAgent


@pytest.fixture
def default_config():
    """Default environment configuration."""
    return {
        'shape': [5, 5],
        'wind_prob': 0.2,
        'reward_step': -1.0,
        'reward_cliff': -50.0,
    }


@pytest.fixture
def no_wind_config():
    """Configuration with no wind (deterministic)."""
    return {
        'shape': [5, 5],
        'wind_prob': 0.0,
        'reward_step': -1.0,
        'reward_cliff': -50.0,
    }


@pytest.fixture
def standard_4x12_config():
    """Standard CliffWalking 4x12 configuration."""
    return {
        'shape': [4, 12],
        'wind_prob': 0.1,
        'reward_step': -1.0,
        'reward_cliff': -100.0,
    }


@pytest.fixture
def seeded_rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def env_factory():
    """Factory function to create environments with custom configs."""
    def _make_env(shape=(5, 5), wind_prob=0.0, reward_cliff=-100.0, reward_step=-1.0):
        env = ResizableCliffWalkingEnv(render_mode=None, shape=tuple(shape))
        env = CliffWalkingWrapper(
            env,
            reward_cliff=reward_cliff,
            reward_step=reward_step,
            terminate_on_cliff=True,
            wind_prob=wind_prob,
        )
        return env
    return _make_env


@pytest.fixture
def cpt_value_func():
    """Default CPT value function with Tversky & Kahneman (1992) parameters."""
    return CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)


@pytest.fixture
def grid_configs():
    """Various grid configurations for parametrized tests."""
    return [
        {'shape': [2, 2], 'name': '2x2'},
        {'shape': [3, 3], 'name': '3x3'},
        {'shape': [4, 12], 'name': '4x12 (standard)'},
        {'shape': [5, 5], 'name': '5x5'},
        {'shape': [10, 10], 'name': '10x10'},
    ]


@pytest.fixture
def wind_configs():
    """Various wind probability configurations."""
    return [0.0, 0.001, 0.1, 0.2, 0.5, 0.999, 1.0]


@pytest.fixture
def torch_seed():
    """Set deterministic seeds for PyTorch reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_env(env_factory):
    """Small deterministic environment for fast agent tests."""
    return env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)


@pytest.fixture
def reinforce_agent(small_env, torch_seed):
    """REINFORCEAgent with deterministic initialization."""
    return REINFORCEAgent(small_env, lr=1e-3, gamma=0.99)


@pytest.fixture
def cpt_reinforce_agent(small_env, torch_seed):
    """CPTREINFORCEAgent with default TK1992 parameters."""
    return CPTREINFORCEAgent(small_env, alpha=0.88, beta=0.88, lambda_=2.25)
