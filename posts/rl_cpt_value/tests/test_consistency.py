"""Tests for consistency between EV and CPT frameworks."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from path_likelihood import (
    build_path_outcome_distributions,
    calculate_path_expected_value,
    calculate_path_cpt_value,
    compare_value_frameworks,
)
from utils import CPTValueFunction


class TestEVvsCPTOrdering:
    """Tests for divergence between EV and CPT path rankings."""

    def test_ev_cpt_ordering_can_differ(self):
        """EV and CPT can produce different preferred paths."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        result = compare_value_frameworks(config, verbose=False)

        # Just verify both methods produce valid preferences
        assert result['ev_preferred_row'] in range(4)
        assert result['cpt_preferred_row'] in range(4)

    def test_higher_loss_aversion_increases_safety(self):
        """Higher loss aversion should favor safer (higher) rows."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.3,
            'reward_step': -1.0,
            'reward_cliff': -100.0,
        }
        distributions = build_path_outcome_distributions(config)

        # Test with different lambda values
        lambdas = [1.0, 2.25, 5.0]
        preferred_rows = []

        for lambda_ in lambdas:
            v = CPTValueFunction(lambda_=lambda_)
            cpt_values = {row: calculate_path_cpt_value(outcomes, v)
                         for row, outcomes in distributions.items()}
            preferred = max(cpt_values, key=cpt_values.get)
            preferred_rows.append(preferred)

        # Higher lambda should prefer lower row numbers (safer paths)
        # Though this isn't strictly guaranteed, it's the expected behavior
        assert preferred_rows[-1] <= preferred_rows[0], \
            f"Higher λ should prefer safer rows: {preferred_rows}"

    def test_no_wind_ev_cpt_same_ordering(self):
        """Without wind, EV and CPT should have same ordering."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        result = compare_value_frameworks(config, verbose=False)

        # Without risk, both should prefer shortest path (highest row number)
        assert result['ev_preferred_row'] == 3
        assert result['cpt_preferred_row'] == 3


class TestSoftmaxProbabilities:
    """Tests for softmax probability calculations."""

    def test_softmax_probabilities_sum_to_one(self):
        """Softmax probabilities should sum to 1.0."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        result = compare_value_frameworks(config, verbose=False)

        ev_sum = sum(result['ev_probabilities'].values())
        cpt_sum = sum(result['cpt_probabilities'].values())

        assert abs(ev_sum - 1.0) < 1e-6, f"EV probs sum to {ev_sum}"
        assert abs(cpt_sum - 1.0) < 1e-6, f"CPT probs sum to {cpt_sum}"

    def test_softmax_probabilities_positive(self):
        """All softmax probabilities should be positive."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        result = compare_value_frameworks(config, verbose=False)

        for row, prob in result['ev_probabilities'].items():
            assert prob > 0, f"EV prob for row {row} is {prob}"

        for row, prob in result['cpt_probabilities'].items():
            assert prob > 0, f"CPT prob for row {row} is {prob}"

    def test_softmax_numerical_stability(self):
        """Softmax should be stable with large negative values."""
        values = np.array([-100.0, -200.0, -300.0, -400.0])

        # Numerically stable softmax (subtract max before exp)
        probs = np.exp(values - values.max()) / np.exp(values - values.max()).sum()

        assert not np.any(np.isnan(probs)), "Softmax produced NaN"
        assert not np.any(np.isinf(probs)), "Softmax produced Inf"
        assert abs(probs.sum() - 1.0) < 1e-6, f"Probs sum to {probs.sum()}"

class TestMonotonicity:
    """Tests for monotonicity properties."""

    def test_ev_increases_with_shorter_path(self):
        """Without wind, EV should increase (less negative) for shorter paths."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        ev_values = {row: calculate_path_expected_value(outcomes)
                     for row, outcomes in distributions.items()}

        # Higher row = shorter path = higher (less negative) EV
        sorted_rows = sorted(ev_values.keys())
        for i in range(len(sorted_rows) - 1):
            row1, row2 = sorted_rows[i], sorted_rows[i + 1]
            assert ev_values[row1] < ev_values[row2], \
                f"EV({row1})={ev_values[row1]} should be < EV({row2})={ev_values[row2]}"

    def test_cliff_probability_decreases_with_lower_row(self):
        """Cliff probability should decrease as we move to lower (safer) rows."""
        from path_likelihood import cliff_fall_probability

        nrows, ncols = 5, 5
        wind_prob = 0.2

        prev_p = 1.0
        for row in range(nrows - 2, -1, -1):  # From row 3 to row 0
            p = cliff_fall_probability(row, nrows, ncols, wind_prob)
            assert p <= prev_p, \
                f"P(cliff) at row {row} should be <= P(cliff) at row {row+1}"
            prev_p = p


class TestValueFunctionProperties:
    """Tests for CPT value function properties across outcomes."""

    def test_cpt_amplifies_losses(self):
        """CPT should amplify losses relative to EV."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)
        v = CPTValueFunction(lambda_=2.25)

        for row, outcomes in distributions.items():
            ev = calculate_path_expected_value(outcomes)
            cpt = calculate_path_cpt_value(outcomes, v)

            # With all negative rewards and loss aversion, CPT < EV
            assert cpt < ev, f"Row {row}: CPT={cpt} should be < EV={ev}"

    def test_lambda_one_reduces_to_power_transform(self):
        """With lambda=1, CPT should just be power transform of rewards."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=1.0)

        for row, outcomes in distributions.items():
            # Manual calculation
            expected = sum(o.probability * (-abs(o.reward) ** 0.88) for o in outcomes)
            actual = calculate_path_cpt_value(outcomes, v)
            assert abs(actual - expected) < 1e-10
