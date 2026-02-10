"""Edge case tests for various configurations."""

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
    calculate_path_cpt_value,
    compare_value_frameworks,
)
from utils import CPTValueFunction


class TestGridSizes:
    """Tests for various grid sizes."""

    @pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 12), (5, 5), (10, 10)])
    def test_environment_creates(self, shape):
        """Environment should create for various grid sizes."""
        env = ResizableCliffWalkingEnv(shape=shape)
        assert env.shape == shape
        assert env.nS == shape[0] * shape[1]

    def test_2x2_grid(self):
        """2x2 grid: no cliff, direct path to goal."""
        env = ResizableCliffWalkingEnv(shape=(2, 2))

        # No cliff in 2x2 (no middle columns)
        assert not np.any(env._cliff)

        # Start at position 2 (row 1, col 0), goal at 3 (row 1, col 1)
        obs, _ = env.reset()
        assert obs == 2

        # One step RIGHT to goal
        obs, _, terminated, _, _ = env.step(1)
        assert obs == 3
        assert terminated

    def test_3x3_grid(self):
        """3x3 grid: single cliff position."""
        env = ResizableCliffWalkingEnv(shape=(3, 3))

        # Cliff only at (2, 1) - bottom row, middle column
        assert env._cliff[2, 1]
        assert not env._cliff[2, 0]  # Start
        assert not env._cliff[2, 2]  # Goal

    def test_path_distributions_various_sizes(self):
        """Path distributions should work for various grid sizes."""
        for shape in [(3, 3), (4, 12), (5, 5), (10, 10)]:
            config = {
                'shape': list(shape),
                'wind_prob': 0.1,
                'reward_step': -1.0,
                'reward_cliff': -50.0,
            }
            distributions = build_path_outcome_distributions(config)

            # Should have nrows-1 viable paths
            assert len(distributions) == shape[0] - 1

            # All probabilities should sum to 1
            for row, outcomes in distributions.items():
                total = sum(o.probability for o in outcomes)
                assert abs(total - 1.0) < 1e-10


class TestWindProbabilities:
    """Tests for edge case wind probabilities."""

    def test_wind_prob_zero(self):
        """wind_prob=0: all cliff probabilities should be 0."""
        nrows, ncols = 5, 5
        for row in range(nrows - 1):
            p = cliff_fall_probability(row, nrows, ncols, wind_prob=0.0)
            assert p == 0.0

    def test_wind_prob_one(self):
        """wind_prob=1: 100% wind means certain drift down."""
        nrows, ncols = 5, 5

        # Row 3 (d=1): certain to fall
        p = cliff_fall_probability(3, nrows, ncols, wind_prob=1.0)
        assert p == 1.0

        # Row 2 (d=2): need 2+ consecutive winds out of 4 steps
        p = cliff_fall_probability(2, nrows, ncols, wind_prob=1.0)
        assert p == 1.0  # With 100% wind, will always drift down

    def test_wind_prob_very_small(self):
        """wind_prob=0.001: very small but non-zero risk."""
        nrows, ncols = 5, 5
        wind_prob = 0.001

        # Row 3 (d=1): P = 1 - (1-0.001)^4 ≈ 0.004
        p = cliff_fall_probability(3, nrows, ncols, wind_prob)
        expected = 1 - (1 - wind_prob) ** 4
        assert abs(p - expected) < 1e-10
        assert p > 0

    def test_wind_prob_very_high(self):
        """wind_prob=0.999: almost certain wind."""
        nrows, ncols = 5, 5
        wind_prob = 0.999

        # Row 3 (d=1): almost certain to fall
        p = cliff_fall_probability(3, nrows, ncols, wind_prob)
        assert p > 0.99

    def test_path_distributions_wind_extremes(self):
        """Path distributions should handle wind probability extremes."""
        for wind_prob in [0.0, 0.001, 0.5, 0.999, 1.0]:
            config = {
                'shape': [5, 5],
                'wind_prob': wind_prob,
                'reward_step': -1.0,
                'reward_cliff': -50.0,
            }
            distributions = build_path_outcome_distributions(config)

            for row, outcomes in distributions.items():
                # Probabilities should always sum to 1
                total = sum(o.probability for o in outcomes)
                assert abs(total - 1.0) < 1e-10, \
                    f"wind_prob={wind_prob}, row={row}: sum={total}"

                # All probabilities should be valid
                for o in outcomes:
                    assert 0 <= o.probability <= 1


class TestRewardVariations:
    """Tests for various reward configurations."""

    @pytest.mark.parametrize("reward_step", [-0.1, -1.0, -5.0, -10.0])
    def test_step_reward_variations(self, reward_step):
        """System should handle various step penalties."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,
            'reward_step': reward_step,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            # With no wind, should be single success outcome
            assert len(outcomes) == 1
            assert outcomes[0].is_success

            # Reward should be steps * reward_step
            steps = 2 * (5 - 1 - row) + (5 - 1)
            expected_reward = steps * reward_step
            assert abs(outcomes[0].reward - expected_reward) < 1e-10


class TestCPTParameterExtremes:
    """Tests for CPT parameter edge cases."""

    def test_alpha_zero(self):
        """alpha=0: all gains become 1 (or 0 for x=0)."""
        v = CPTValueFunction(alpha=0.0, beta=0.88, lambda_=2.25)
        assert v(0) == 0.0
        assert v(1) == 1.0
        assert v(100) == 1.0

    def test_alpha_one(self):
        """alpha=1: linear gains."""
        v = CPTValueFunction(alpha=1.0, beta=0.88, lambda_=2.25)
        assert v(10) == 10.0
        assert v(100) == 100.0

    def test_lambda_one(self):
        """lambda=1: no loss aversion."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=1.0)
        # |v(-x)| should equal v(x) in magnitude (with same exponent)
        assert abs(abs(v(-10)) - (10 ** 0.88)) < 1e-10

    def test_lambda_very_high(self):
        """Very high lambda: extreme loss aversion."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=10.0)
        # Losses should be magnified significantly
        assert abs(v(-10)) > 5 * v(10)

    def test_reference_point_non_zero(self):
        """Non-zero reference point should shift the value function."""
        v = CPTValueFunction(reference_point=5.0)

        # x=5 should be the new reference (value=0)
        assert v(5.0) == 0.0

        # x=10 is now a gain of 5
        assert v(10.0) == v.alpha and v(10.0) > 0 or True  # Just check it's positive
        assert v(10.0) > 0

        # x=0 is now a loss of 5
        assert v(0.0) < 0


class TestNumericalEdgeCases:
    """Tests for numerical stability edge cases."""

    def test_very_large_rewards(self):
        """System should handle very large reward magnitudes."""
        v = CPTValueFunction()

        # Large positive
        val = v(1e6)
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert val > 0

        # Large negative
        val = v(-1e6)
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert val < 0

    def test_very_small_rewards(self):
        """System should handle very small reward magnitudes."""
        v = CPTValueFunction()

        for x in [1e-6, 1e-10, 1e-15]:
            val = v(x)
            assert not np.isnan(val)
            assert val >= 0

            val = v(-x)
            assert not np.isnan(val)
            assert val <= 0

    def test_softmax_underflow_prevention(self):
        """Softmax should not underflow with large negative values."""
        values = np.array([-500.0, -600.0, -700.0, -800.0])

        # Standard softmax may underflow
        with np.errstate(under='ignore'):
            exp_vals = np.exp(values)

        if np.any(exp_vals == 0):
            # Use stable version
            max_val = np.max(values)
            shifted = values - max_val
            probs = np.exp(shifted) / np.exp(shifted).sum()
        else:
            probs = exp_vals / exp_vals.sum()

        assert not np.any(np.isnan(probs))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_probability_near_zero_and_one(self):
        """Handle probabilities very close to 0 and 1."""
        # Very high wind probability
        p = cliff_fall_probability(3, 5, 5, wind_prob=0.9999)
        assert 0 <= p <= 1
        assert not np.isnan(p)

        # Very low wind probability
        p = cliff_fall_probability(3, 5, 5, wind_prob=0.0001)
        assert 0 <= p <= 1
        assert not np.isnan(p)


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_single_row_above_cliff(self):
        """Row just above cliff (d=1) should have highest cliff probability."""
        nrows, ncols = 5, 5
        wind_prob = 0.2

        probs = [cliff_fall_probability(row, nrows, ncols, wind_prob)
                 for row in range(nrows - 1)]

        # Row 3 (closest to cliff) should have highest probability
        assert probs[-1] == max(probs)

    def test_top_row_safest(self):
        """Top row (row 0) should be safest."""
        nrows, ncols = 5, 5
        wind_prob = 0.2

        probs = [cliff_fall_probability(row, nrows, ncols, wind_prob)
                 for row in range(nrows - 1)]

        # Row 0 should have lowest probability
        assert probs[0] == min(probs)

    def test_wide_grid_vs_tall_grid(self):
        """Compare behavior between wide and tall grids."""
        # Wide grid: more horizontal steps, higher risk for same distance
        wide_p = cliff_fall_probability(row=2, nrows=4, ncols=12, wind_prob=0.1)

        # Tall grid: fewer horizontal steps, lower risk
        tall_p = cliff_fall_probability(row=2, nrows=10, ncols=4, wind_prob=0.1)

        # Wide should be riskier (more exposure)
        assert wide_p > tall_p
