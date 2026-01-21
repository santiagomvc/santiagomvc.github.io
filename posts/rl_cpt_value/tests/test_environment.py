"""Tests for CliffWalking environment mechanics."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_cliff_walking import ResizableCliffWalkingEnv, CliffWalkingWrapper


class TestStateSpace:
    """Tests for state space verification."""

    @pytest.mark.parametrize("shape", [(3, 3), (4, 12), (5, 5), (10, 10)])
    def test_state_space_size(self, shape):
        """State space size should be nrows * ncols."""
        env = ResizableCliffWalkingEnv(shape=shape)
        expected_nS = shape[0] * shape[1]
        assert env.nS == expected_nS
        assert env.observation_space.n == expected_nS

    @pytest.mark.parametrize("shape", [(3, 3), (4, 12), (5, 5)])
    def test_start_position(self, shape):
        """Start position should be bottom-left."""
        env = ResizableCliffWalkingEnv(shape=shape)
        nrows, ncols = shape
        expected_start = (nrows - 1) * ncols  # Bottom-left corner
        assert env.start_state_index == expected_start

        obs, _ = env.reset()
        assert obs == expected_start

    @pytest.mark.parametrize("shape", [(3, 3), (4, 12), (5, 5)])
    def test_goal_position(self, shape):
        """Goal should be bottom-right."""
        env = ResizableCliffWalkingEnv(shape=shape)
        nrows, ncols = shape
        goal_state = nrows * ncols - 1  # Bottom-right corner

        # Verify goal is not on cliff
        goal_row = goal_state // ncols
        goal_col = goal_state % ncols
        assert goal_row == nrows - 1
        assert goal_col == ncols - 1
        assert not env._cliff[goal_row, goal_col]


class TestCliffPositions:
    """Tests for cliff position verification."""

    @pytest.mark.parametrize("shape", [(3, 3), (4, 12), (5, 5)])
    def test_cliff_positions(self, shape):
        """Cliff should be bottom row between start and goal."""
        env = ResizableCliffWalkingEnv(shape=shape)
        nrows, ncols = shape

        # Start position (col 0) should not be cliff
        assert not env._cliff[nrows - 1, 0]

        # Goal position (col ncols-1) should not be cliff
        assert not env._cliff[nrows - 1, ncols - 1]

        # Middle columns of bottom row should be cliff (if ncols > 2)
        if ncols > 2:
            for col in range(1, ncols - 1):
                assert env._cliff[nrows - 1, col], f"Position ({nrows-1}, {col}) should be cliff"

        # All other rows should not have cliff
        for row in range(nrows - 1):
            for col in range(ncols):
                assert not env._cliff[row, col], f"Position ({row}, {col}) should not be cliff"

    def test_cliff_2x2_grid(self):
        """2x2 grid has no cliff positions (no middle columns)."""
        env = ResizableCliffWalkingEnv(shape=(2, 2))
        assert not np.any(env._cliff)


class TestTransitionProbabilities:
    """Tests for transition probability consistency."""

    @pytest.mark.parametrize("shape", [(3, 3), (4, 12), (5, 5)])
    def test_transition_probabilities_sum_to_one(self, shape):
        """Transition probabilities should sum to 1.0 for each state-action pair."""
        env = ResizableCliffWalkingEnv(shape=shape)

        for state in range(env.nS):
            for action in range(env.nA):
                probs = [p for p, _, _, _ in env.P[state][action]]
                total = sum(probs)
                assert abs(total - 1.0) < 1e-10, \
                    f"State {state}, action {action}: probs sum to {total}"

    @pytest.mark.parametrize("shape", [(3, 3), (5, 5)])
    def test_deterministic_transitions(self, shape):
        """Non-slippery environment should have deterministic transitions."""
        env = ResizableCliffWalkingEnv(shape=shape, is_slippery=False)

        for state in range(env.nS):
            for action in range(env.nA):
                transitions = env.P[state][action]
                assert len(transitions) == 1, \
                    f"State {state}, action {action}: expected 1 transition, got {len(transitions)}"
                prob, _, _, _ = transitions[0]
                assert prob == 1.0


class TestWindMechanics:
    """Tests for wind mechanics validation."""

    def test_wind_modifies_action(self, env_factory):
        """Wind should sometimes override action to DOWN."""
        env = env_factory(shape=(5, 5), wind_prob=1.0)  # 100% wind
        env.reset(seed=42)

        # Start at bottom-left (row 4), take UP action
        # With 100% wind, should stay in place (can't go past bottom)
        obs, _ = env.reset()
        start_row = obs // 5

        # Take many steps with 100% wind
        for _ in range(10):
            obs, _, _, _, _ = env.step(0)  # UP action
            row = obs // 5
            # Should never go up because wind always pushes DOWN
            assert row >= start_row - 1  # Allow at most 1 row up (due to initial position)

    def test_no_wind_deterministic(self, env_factory):
        """With wind_prob=0, actions should be deterministic."""
        env = env_factory(shape=(5, 5), wind_prob=0.0)
        env.reset(seed=42)

        # Start at bottom-left, go UP
        obs, _ = env.reset()
        start_row = obs // 5

        obs, _, _, _, _ = env.step(0)  # UP
        new_row = obs // 5
        assert new_row == start_row - 1, "UP action should move one row up"

    def test_wind_probability_statistical(self, env_factory):
        """Wind should occur at approximately wind_prob rate."""
        wind_prob = 0.3
        env = env_factory(shape=(10, 10), wind_prob=wind_prob)
        env.reset(seed=42)

        # Start from a safe position and track wind effects
        n_trials = 1000
        wind_occurred = 0

        for i in range(n_trials):
            env.reset(seed=i)
            # Try to go UP from start (row 9)
            obs, _, _, _, _ = env.step(0)  # UP action
            row = obs // 10
            # If still at row 9, wind pushed us back down (or we were blocked)
            if row == 9:
                wind_occurred += 1

        # Allow 10% deviation from expected rate
        expected = n_trials * wind_prob
        assert abs(wind_occurred - expected) < 0.1 * n_trials, \
            f"Wind occurred {wind_occurred} times, expected ~{expected}"


class TestRewardSignals:
    """Tests for reward signal verification."""

    def test_step_reward(self, env_factory):
        """Normal step should give step reward."""
        reward_step = -1.5
        env = env_factory(shape=(5, 5), wind_prob=0.0, reward_step=reward_step)
        env.reset(seed=42)

        # Take a safe step (UP from start)
        _, reward, _, _, _ = env.step(0)
        assert reward == reward_step

    def test_cliff_reward(self, env_factory):
        """Falling into cliff should give cliff reward and terminate."""
        reward_cliff = -100.0
        env = env_factory(shape=(5, 5), wind_prob=0.0, reward_cliff=reward_cliff)
        env.reset(seed=42)

        # Move RIGHT from start (into cliff)
        _, reward, terminated, _, _ = env.step(1)
        assert reward == reward_cliff
        assert terminated

    def test_goal_terminates(self, env_factory):
        """Reaching goal should terminate episode."""
        env = env_factory(shape=(3, 3), wind_prob=0.0)
        env.reset(seed=42)

        # 3x3 grid: path is UP, RIGHT, RIGHT, DOWN
        env.step(0)  # UP to row 1
        env.step(1)  # RIGHT to col 1
        env.step(1)  # RIGHT to col 2
        _, _, terminated, _, _ = env.step(2)  # DOWN to goal

        assert terminated


class TestEnvironmentReset:
    """Tests for environment reset behavior."""

    def test_reset_returns_start_state(self, env_factory):
        """Reset should always return start state."""
        env = env_factory(shape=(5, 5))

        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            expected_start = 4 * 5  # Row 4, col 0
            assert obs == expected_start

    def test_reset_after_cliff_fall(self, env_factory):
        """After cliff fall, reset should work correctly."""
        env = env_factory(shape=(5, 5), wind_prob=0.0)
        env.reset(seed=42)

        # Fall into cliff
        env.step(1)  # RIGHT into cliff

        # Reset should return to start
        obs, _ = env.reset()
        expected_start = 4 * 5
        assert obs == expected_start

    def test_reset_seed_reproducibility(self, env_factory):
        """Same seed should produce same sequence."""
        env = env_factory(shape=(5, 5), wind_prob=0.5)

        # Run 1
        env.reset(seed=42)
        rewards1 = []
        for _ in range(10):
            _, r, term, _, _ = env.step(0)
            rewards1.append(r)
            if term:
                break

        # Run 2 with same seed
        env.reset(seed=42)
        rewards2 = []
        for _ in range(10):
            _, r, term, _, _ = env.step(0)
            rewards2.append(r)
            if term:
                break

        assert rewards1 == rewards2
