"""Tests for mathematical formulas and calculations."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from path_likelihood import (
    cliff_fall_probability,
    build_path_outcome_distributions,
    calculate_path_expected_value,
    calculate_path_cpt_value,
    PathOutcome,
)
from utils import CPTValueFunction, CPTWeightingFunction, SlidingWindowCPT


class TestStepCounting:
    """Tests for step counting formula validation."""

    @pytest.mark.parametrize("nrows,ncols", [(3, 3), (4, 12), (5, 5), (10, 10)])
    def test_step_counting_formula(self, nrows, ncols):
        """Formula 2*(nrows-1-row) + (ncols-1) should match manual calculation."""
        for row in range(nrows - 1):
            # Manual calculation: UP + RIGHT + DOWN
            up_steps = nrows - 1 - row      # From start row (nrows-1) to target row
            right_steps = ncols - 1         # Across the grid
            down_steps = nrows - 1 - row    # From target row back to goal row (nrows-1)
            expected = up_steps + right_steps + down_steps

            # Formula from code
            formula = 2 * (nrows - 1 - row) + (ncols - 1)

            assert formula == expected, \
                f"Grid {nrows}x{ncols}, row {row}: formula={formula}, expected={expected}"

    def test_step_counting_5x5_detailed(self):
        """Detailed step count verification for 5x5 grid."""
        nrows, ncols = 5, 5
        expected_steps = {
            0: 12,  # UP(4) + RIGHT(4) + DOWN(4)
            1: 10,  # UP(3) + RIGHT(4) + DOWN(3)
            2: 8,   # UP(2) + RIGHT(4) + DOWN(2)
            3: 6,   # UP(1) + RIGHT(4) + DOWN(1)
        }

        config = {'shape': [nrows, ncols], 'wind_prob': 0.0, 'reward_step': -1.0, 'reward_cliff': -50.0}
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            # Success outcome has total steps * reward_step
            success_outcome = next(o for o in outcomes if o.is_success)
            actual_steps = int(-success_outcome.reward)  # reward_step = -1
            assert actual_steps == expected_steps[row], \
                f"Row {row}: got {actual_steps} steps, expected {expected_steps[row]}"


class TestCliffProbability:
    """Tests for cliff probability calculations."""

    def test_cliff_row_probability_is_one(self):
        """Being on cliff row (d=0) should have P(cliff)=1."""
        p = cliff_fall_probability(row=4, nrows=5, ncols=5, wind_prob=0.5)
        assert p == 1.0

    def test_no_wind_probability_is_zero(self):
        """With wind_prob=0, P(cliff) should be 0 for rows above cliff."""
        for row in range(4):
            p = cliff_fall_probability(row, nrows=5, ncols=5, wind_prob=0.0)
            assert p == 0.0, f"Row {row} should have P(cliff)=0 with no wind"

    def test_distance_greater_than_steps(self):
        """If distance to cliff >= horizontal steps, P(cliff)=0."""
        # 3x3 grid: 2 horizontal steps, row 0 has distance 2
        p = cliff_fall_probability(row=0, nrows=3, ncols=3, wind_prob=0.5)
        assert p == 0.0, "Can't fall if distance >= horizontal steps"

    def test_distance_one_exact_formula(self):
        """Distance=1: P(cliff) = 1 - (1-wind_prob)^h."""
        nrows, ncols = 5, 5
        row = 3  # Distance = 1
        h = ncols - 1

        for wind_prob in [0.1, 0.2, 0.5]:
            expected = 1 - (1 - wind_prob) ** h
            actual = cliff_fall_probability(row, nrows, ncols, wind_prob)
            assert abs(actual - expected) < 1e-10, \
                f"wind_prob={wind_prob}: got {actual}, expected {expected}"

    def test_distance_two_approximation(self):
        """Distance>=2: Uses union bound approximation (h-d+1)*wind_prob^d."""
        nrows, ncols = 5, 5
        row = 2  # Distance = 2
        wind_prob = 0.2
        h = ncols - 1
        d = 2

        expected = (h - d + 1) * (wind_prob ** d)
        actual = cliff_fall_probability(row, nrows, ncols, wind_prob)
        assert abs(actual - expected) < 1e-10

    def test_probability_bounded_by_one(self):
        """Probability should never exceed 1.0."""
        # High wind probability could make union bound > 1
        p = cliff_fall_probability(row=2, nrows=5, ncols=10, wind_prob=0.8)
        assert p <= 1.0


class TestCPTValueFunction:
    """Tests for CPT value function implementation."""

    def test_reference_point(self):
        """v(reference_point) should equal 0."""
        v = CPTValueFunction(reference_point=0.0)
        assert v(0) == 0.0

        v = CPTValueFunction(reference_point=10.0)
        assert v(10.0) == 0.0

    def test_gains_formula(self):
        """v(x) = x^alpha for x >= 0."""
        alpha = 0.88
        v = CPTValueFunction(alpha=alpha)

        test_values = [1, 10, 50, 100]
        for x in test_values:
            expected = x ** alpha
            assert abs(v(x) - expected) < 1e-10, f"v({x}) = {v(x)}, expected {expected}"

    def test_losses_formula(self):
        """v(x) = -lambda * (-x)^beta for x < 0."""
        beta = 0.88
        lambda_ = 2.25
        v = CPTValueFunction(beta=beta, lambda_=lambda_)

        test_values = [1, 10, 50, 100]
        for x in test_values:
            expected = -lambda_ * (x ** beta)
            assert abs(v(-x) - expected) < 1e-10, f"v({-x}) = {v(-x)}, expected {expected}"

    def test_loss_aversion(self):
        """Loss aversion: |v(-x)| > v(x) for x > 0."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)

        test_values = [1, 10, 50, 100]
        for x in test_values:
            gain_value = v(x)
            loss_magnitude = abs(v(-x))
            assert loss_magnitude > gain_value, \
                f"|v({-x})|={loss_magnitude} should be > v({x})={gain_value}"

    def test_diminishing_sensitivity(self):
        """Both gains and losses should show diminishing sensitivity."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)

        # For gains: marginal value of x should decrease
        # v(20) - v(10) < v(10) - v(0)
        margin_10_20 = v(20) - v(10)
        margin_0_10 = v(10) - v(0)
        assert margin_10_20 < margin_0_10, "Diminishing sensitivity for gains"

        # For losses: marginal value should also decrease (in magnitude)
        margin_neg20_neg10 = abs(v(-10) - v(-20))
        margin_neg10_0 = abs(v(0) - v(-10))
        assert margin_neg20_neg10 < margin_neg10_0, "Diminishing sensitivity for losses"

    def test_tversky_kahneman_1992_values(self):
        """Test against Tversky & Kahneman (1992) parameter values."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)

        # v(10) = 10^0.88 ≈ 7.586
        assert abs(v(10) - 10 ** 0.88) < 1e-10

        # v(-10) = -2.25 * 10^0.88 ≈ -17.07
        assert abs(v(-10) - (-2.25 * 10 ** 0.88)) < 1e-10


class TestExpectedValueCalculation:
    """Tests for expected value calculation."""

    def test_deterministic_ev(self):
        """With no wind, EV should equal the deterministic outcome."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            ev = calculate_path_expected_value(outcomes)
            # Should have exactly one outcome with p=1.0
            assert len(outcomes) == 1
            assert outcomes[0].probability == 1.0
            assert ev == outcomes[0].reward

    def test_ev_probability_weighted(self):
        """EV should be probability-weighted average."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            # Manual calculation
            expected_ev = sum(o.probability * o.reward for o in outcomes)
            actual_ev = calculate_path_expected_value(outcomes)
            assert abs(actual_ev - expected_ev) < 1e-10


class TestCPTValueCalculation:
    """Tests for CPT value calculation."""

    def test_cpt_applies_value_function(self):
        """CPT value should apply v() before probability weighting."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)
        v = CPTValueFunction()

        for row, outcomes in distributions.items():
            # Manual calculation
            expected_cpt = sum(o.probability * v(o.reward) for o in outcomes)
            actual_cpt = calculate_path_cpt_value(outcomes, v)
            assert abs(actual_cpt - expected_cpt) < 1e-10

    def test_cpt_more_negative_than_ev(self):
        """CPT value should be more negative than EV due to loss aversion."""
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
            # All rewards are negative, so CPT amplifies the loss
            assert cpt < ev, f"Row {row}: CPT={cpt} should be < EV={ev}"


class TestProbabilityNormalization:
    """Tests for probability normalization in outcome distributions."""

    @pytest.mark.parametrize("wind_prob", [0.0, 0.1, 0.2, 0.5, 0.8])
    def test_probabilities_sum_to_one(self, wind_prob):
        """Outcome probabilities should sum to 1.0."""
        config = {
            'shape': [5, 5],
            'wind_prob': wind_prob,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            total = sum(o.probability for o in outcomes)
            assert abs(total - 1.0) < 1e-10, \
                f"Row {row}, wind={wind_prob}: probabilities sum to {total}"

    def test_success_and_cliff_probabilities(self):
        """Success and cliff probabilities should be complementary."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            p_success = sum(o.probability for o in outcomes if o.is_success)
            p_cliff = sum(o.probability for o in outcomes if not o.is_success)
            assert abs(p_success + p_cliff - 1.0) < 1e-10


class TestCPTWeightingFunction:
    """Tests for CPT probability weighting function."""

    def test_weighting_function_boundaries(self):
        """w(0) = 0 and w(1) = 1 for both w+ and w-."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        assert w.w_plus(0) == 0.0
        assert w.w_plus(1) == 1.0
        assert w.w_minus(0) == 0.0
        assert w.w_minus(1) == 1.0

    def test_weighting_function_inverse_s(self):
        """Inverse S-shape: w(p) > p for small p, w(p) < p for large p."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        # Small probabilities should be overweighted
        for p in [0.01, 0.05, 0.1]:
            assert w.w_plus(p) > p, f"w+(p={p}) should overweight small p"
            assert w.w_minus(p) > p, f"w-(p={p}) should overweight small p"

        # Large probabilities should be underweighted
        for p in [0.8, 0.9, 0.95]:
            assert w.w_plus(p) < p, f"w+(p={p}) should underweight large p"
            assert w.w_minus(p) < p, f"w-(p={p}) should underweight large p"

    def test_weighting_function_crossover(self):
        """w(0.5) < 0.5 — subcertainty from Tversky & Kahneman 1992."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        assert w.w_plus(0.5) < 0.5, "w+(0.5) should be < 0.5 (subcertainty)"
        assert w.w_minus(0.5) < 0.5, "w-(0.5) should be < 0.5 (subcertainty)"

    def test_weighting_function_monotonic(self):
        """w(p) should be monotonically increasing."""
        w = CPTWeightingFunction()
        probs = [i / 100 for i in range(101)]
        w_plus_vals = [w.w_plus(p) for p in probs]
        w_minus_vals = [w.w_minus(p) for p in probs]

        for i in range(len(probs) - 1):
            assert w_plus_vals[i] <= w_plus_vals[i + 1], \
                f"w+ not monotonic at p={probs[i]}"
            assert w_minus_vals[i] <= w_minus_vals[i + 1], \
                f"w- not monotonic at p={probs[i]}"

    def test_weighting_function_gamma_one_is_identity(self):
        """With gamma=1.0, w(p) should equal p (no distortion)."""
        w = CPTWeightingFunction(gamma_plus=1.0, gamma_minus=1.0)
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert abs(w.w_plus(p) - p) < 1e-10, f"w+(p={p}) should equal p when gamma=1"
            assert abs(w.w_minus(p) - p) < 1e-10, f"w-(p={p}) should equal p when gamma=1"


class TestDecisionWeights:
    """Tests for CPT decision weights via SlidingWindowCPT."""

    def test_decision_weights_pure_gains(self):
        """Decision weights for pure gains should sum to n (normalized)."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        sw = SlidingWindowCPT(w, max_batches=5, decay=0.8, reference_point=0.0)

        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        sw.add_batch(returns)
        weights = sw.compute_decision_weights(returns)

        assert abs(weights.sum() - len(returns)) < 1e-6, \
            f"Weights should sum to n={len(returns)}, got {weights.sum()}"

    def test_decision_weights_pure_losses(self):
        """Decision weights for pure losses should sum to n (normalized)."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        sw = SlidingWindowCPT(w, max_batches=5, decay=0.8, reference_point=0.0)

        returns = [-5.0, -4.0, -3.0, -2.0, -1.0]
        sw.add_batch(returns)
        weights = sw.compute_decision_weights(returns)

        assert abs(weights.sum() - len(returns)) < 1e-6, \
            f"Weights should sum to n={len(returns)}, got {weights.sum()}"

    def test_rare_events_overweighted(self):
        """Rare extreme events should get higher decision weights than common ones."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        sw = SlidingWindowCPT(w, max_batches=5, decay=0.8, reference_point=0.0)

        # Most returns are moderate, one is extreme
        returns = [10.0] * 7 + [100.0]  # 100.0 is rare
        sw.add_batch(returns)
        weights = sw.compute_decision_weights(returns)

        # The rare extreme return should have higher weight per-instance
        # than the common moderate returns
        rare_weight = weights[7]  # 100.0
        common_weight = weights[0]  # 10.0 (one of many)
        assert rare_weight > common_weight, \
            f"Rare event weight ({rare_weight}) should exceed common ({common_weight})"

    def test_sliding_window_decay(self):
        """Old batches should have reduced weight after decay."""
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        sw = SlidingWindowCPT(w, max_batches=5, decay=0.5, reference_point=0.0)

        sw.add_batch([1.0, 2.0, 3.0])
        # After adding second batch, first batch weight should be decayed
        sw.add_batch([4.0, 5.0, 6.0])

        assert len(sw.buffer) == 2
        # First batch weight should be 0.5 (decayed once)
        assert abs(sw.buffer[0][1] - 0.5) < 1e-10
        # Second batch weight should be 1.0 (just added)
        assert abs(sw.buffer[1][1] - 1.0) < 1e-10

    def test_sliding_window_max_batches(self):
        """Buffer should not exceed max_batches."""
        w = CPTWeightingFunction()
        sw = SlidingWindowCPT(w, max_batches=3, decay=0.8)

        for i in range(5):
            sw.add_batch([float(i)])

        assert len(sw.buffer) == 3

    def test_cpt_value_with_weighting_differs(self):
        """Full CPT value (with weighting) should differ from value-only CPT."""
        outcomes = [
            PathOutcome(reward=-6.0, probability=0.9, is_success=True, description="safe"),
            PathOutcome(reward=-56.0, probability=0.1, is_success=False, description="cliff"),
        ]
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)
        w = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        cpt_no_weighting = calculate_path_cpt_value(outcomes, v)
        cpt_with_weighting = calculate_path_cpt_value(outcomes, v, w)

        assert cpt_no_weighting != cpt_with_weighting, \
            "CPT with probability weighting should differ from value-only CPT"
