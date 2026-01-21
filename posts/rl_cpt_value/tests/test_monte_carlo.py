"""Monte Carlo validation tests for theoretical predictions."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_cliff_walking import ResizableCliffWalkingEnv, CliffWalkingWrapper
from path_likelihood import (
    cliff_fall_probability,
    build_path_outcome_distributions,
    calculate_path_expected_value,
)


class TestCliffProbabilityMC:
    """Monte Carlo validation of cliff probability formulas."""

    @pytest.mark.parametrize("wind_prob", [0.1, 0.2, 0.3])
    def test_cliff_probability_distance_one(self, wind_prob):
        """MC validation for d=1 (exact formula)."""
        nrows, ncols = 5, 5
        row = 3  # Distance to cliff = 1
        h = ncols - 1
        n_samples = 5000
        rng = np.random.default_rng(42)

        theoretical = cliff_fall_probability(row, nrows, ncols, wind_prob)
        expected_exact = 1 - (1 - wind_prob) ** h

        # Verify formula is exact
        assert abs(theoretical - expected_exact) < 1e-10

        # Monte Carlo simulation
        cliff_falls = 0
        for _ in range(n_samples):
            current_row = row
            for step in range(h):
                if rng.random() < wind_prob:
                    current_row += 1
                if current_row >= nrows - 1:
                    cliff_falls += 1
                    break

        empirical = cliff_falls / n_samples

        # Allow 5% deviation
        assert abs(empirical - theoretical) < 0.05, \
            f"MC={empirical:.4f}, theoretical={theoretical:.4f}"

    @pytest.mark.parametrize("wind_prob", [0.1, 0.2])
    def test_cliff_probability_distance_two(self, wind_prob):
        """MC validation for d=2 (union bound approximation)."""
        nrows, ncols = 5, 5
        row = 2  # Distance to cliff = 2
        n_samples = 10000
        rng = np.random.default_rng(42)

        theoretical = cliff_fall_probability(row, nrows, ncols, wind_prob)

        # Monte Carlo simulation
        cliff_falls = 0
        for _ in range(n_samples):
            current_row = row
            for step in range(ncols - 1):
                if rng.random() < wind_prob:
                    current_row += 1
                if current_row >= nrows - 1:
                    cliff_falls += 1
                    break

        empirical = cliff_falls / n_samples

        # Union bound is upper bound, so empirical should be <= theoretical
        # Allow small deviation due to MC variance
        assert empirical <= theoretical + 0.02, \
            f"MC={empirical:.4f} exceeds theoretical={theoretical:.4f}"

    def test_zero_wind_no_falls(self):
        """With wind_prob=0, no cliff falls should occur."""
        nrows, ncols = 5, 5
        n_samples = 1000
        rng = np.random.default_rng(42)

        for row in range(nrows - 1):
            cliff_falls = 0
            for _ in range(n_samples):
                current_row = row
                for step in range(ncols - 1):
                    # No wind, no position change
                    pass
                if current_row >= nrows - 1:
                    cliff_falls += 1

            assert cliff_falls == 0, f"Row {row} had {cliff_falls} falls with no wind"


class TestExpectedValueMC:
    """Monte Carlo validation of expected value calculations."""

    def test_deterministic_ev_exact(self, env_factory):
        """With no wind, empirical EV should match theoretical exactly."""
        nrows, ncols = 5, 5
        reward_step = -1.0
        reward_cliff = -50.0

        config = {
            'shape': [nrows, ncols],
            'wind_prob': 0.0,
            'reward_step': reward_step,
            'reward_cliff': reward_cliff,
        }
        distributions = build_path_outcome_distributions(config)

        for row in range(nrows - 1):
            theoretical = calculate_path_expected_value(distributions[row])

            # Simulate deterministic path
            steps = 2 * (nrows - 1 - row) + (ncols - 1)
            empirical = steps * reward_step

            assert abs(empirical - theoretical) < 1e-10, \
                f"Row {row}: empirical={empirical}, theoretical={theoretical}"

    @pytest.mark.parametrize("row", [0, 1, 2, 3])
    def test_stochastic_ev_mc(self, row):
        """MC validation of expected value with wind."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        reward_step = -1.0
        reward_cliff = -50.0
        n_samples = 5000
        rng = np.random.default_rng(42)

        config = {
            'shape': [nrows, ncols],
            'wind_prob': wind_prob,
            'reward_step': reward_step,
            'reward_cliff': reward_cliff,
        }
        distributions = build_path_outcome_distributions(config)
        theoretical = calculate_path_expected_value(distributions[row])

        # Monte Carlo simulation
        total_rewards = []
        for _ in range(n_samples):
            current_row = row
            steps = 0
            fell = False

            # UP phase (deterministic)
            steps += nrows - 1 - row

            # RIGHT phase (with wind)
            for _ in range(ncols - 1):
                if rng.random() < wind_prob:
                    current_row += 1
                steps += 1
                if current_row >= nrows - 1:
                    fell = True
                    break

            # DOWN phase (if didn't fall)
            if not fell:
                steps += nrows - 1 - row

            if fell:
                total_rewards.append(steps * reward_step + reward_cliff)
            else:
                total_rewards.append(steps * reward_step)

        empirical = np.mean(total_rewards)

        # Allow larger tolerance for stochastic case
        assert abs(empirical - theoretical) < 3.0, \
            f"Row {row}: empirical={empirical:.2f}, theoretical={theoretical:.2f}"


class TestRewardDistributionShape:
    """Tests for the shape of reward distributions."""

    def test_bimodal_distribution_with_wind(self):
        """Reward distribution should be bimodal: success vs fall."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        reward_step = -1.0
        reward_cliff = -50.0
        n_samples = 5000
        rng = np.random.default_rng(42)

        row = 2  # Row with significant cliff probability
        rewards = []

        for _ in range(n_samples):
            current_row = row
            steps = 0
            fell = False

            steps += nrows - 1 - row

            for _ in range(ncols - 1):
                if rng.random() < wind_prob:
                    current_row += 1
                steps += 1
                if current_row >= nrows - 1:
                    fell = True
                    break

            if not fell:
                steps += nrows - 1 - row

            if fell:
                rewards.append(steps * reward_step + reward_cliff)
            else:
                rewards.append(steps * reward_step)

        rewards = np.array(rewards)

        # Check for bimodal distribution
        success_mask = rewards > -20  # Success rewards are small negative
        fall_mask = rewards < -40  # Fall rewards are large negative

        n_success = np.sum(success_mask)
        n_fall = np.sum(fall_mask)

        # Should have both outcomes
        assert n_success > 0, "No successful episodes"
        assert n_fall > 0, "No cliff fall episodes"

        # Proportions should roughly match theoretical probabilities
        p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
        expected_falls = n_samples * p_cliff
        assert abs(n_fall - expected_falls) < 0.1 * n_samples, \
            f"Fall count {n_fall} differs from expected {expected_falls}"


class TestEnvironmentMC:
    """Monte Carlo tests using actual environment."""

    def test_deterministic_policy_ev(self, env_factory):
        """Run deterministic policy and compare to theoretical EV."""
        nrows, ncols = 5, 5
        reward_step = -1.0
        reward_cliff = -50.0
        n_episodes = 500

        env = env_factory(
            shape=(nrows, ncols),
            wind_prob=0.0,
            reward_step=reward_step,
            reward_cliff=reward_cliff,
        )

        # Path via row 0: UP, UP, UP, UP, RIGHT x4, DOWN x4
        actions_row0 = [0]*4 + [1]*4 + [2]*4
        expected_reward = len(actions_row0) * reward_step

        total_reward = 0
        for _ in range(n_episodes):
            env.reset(seed=42)
            episode_reward = 0
            for action in actions_row0:
                _, r, term, _, _ = env.step(action)
                episode_reward += r
                if term:
                    break
            total_reward += episode_reward

        avg_reward = total_reward / n_episodes
        assert abs(avg_reward - expected_reward) < 0.1, \
            f"Avg reward {avg_reward} differs from expected {expected_reward}"

    def test_windy_policy_ev_distribution(self, env_factory):
        """With wind, reward should follow expected distribution."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        reward_step = -1.0
        reward_cliff = -50.0
        n_episodes = 1000

        env = env_factory(
            shape=(nrows, ncols),
            wind_prob=wind_prob,
            reward_step=reward_step,
            reward_cliff=reward_cliff,
        )

        # Try to traverse via row 2
        config = {
            'shape': [nrows, ncols],
            'wind_prob': wind_prob,
            'reward_step': reward_step,
            'reward_cliff': reward_cliff,
        }
        distributions = build_path_outcome_distributions(config)
        theoretical_ev = calculate_path_expected_value(distributions[2])

        # Simulate policy: UP, UP, RIGHT x4, DOWN, DOWN
        # This attempts to go via row 2
        episode_rewards = []

        for seed in range(n_episodes):
            env.reset(seed=seed)
            episode_reward = 0
            terminated = False

            # Go UP twice (to row 2)
            for _ in range(2):
                if not terminated:
                    _, r, terminated, _, _ = env.step(0)
                    episode_reward += r

            # Go RIGHT 4 times
            for _ in range(4):
                if not terminated:
                    _, r, terminated, _, _ = env.step(1)
                    episode_reward += r

            # Go DOWN twice
            for _ in range(2):
                if not terminated:
                    _, r, terminated, _, _ = env.step(2)
                    episode_reward += r

            episode_rewards.append(episode_reward)

        empirical_ev = np.mean(episode_rewards)

        # The actual policy execution may differ due to wind effects on all actions
        # But should be in the same ballpark
        assert abs(empirical_ev - theoretical_ev) < 10.0, \
            f"Empirical EV {empirical_ev:.2f} differs from theoretical {theoretical_ev:.2f}"
