"""Pytest fixtures for CPT CliffWalking diagnostics."""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_cliff_walking import ResizableCliffWalkingEnv, CliffWalkingWrapper
from utils import CPTValueFunction


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
