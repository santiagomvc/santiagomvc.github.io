"""Theoretical tests for CPT implementation correctness."""

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


class TestValueOnlyCPT:
    """Tests verifying value-only CPT implementation (no probability weighting)."""

    def test_no_probability_distortion(self):
        """Verify CPT uses objective probabilities, not distorted ones."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)
        v = CPTValueFunction()

        for row, outcomes in distributions.items():
            # CPT calculation: Σ p_i * v(r_i)
            # NOT the full CPT: Σ w(p_i) * v(r_i)
            cpt = calculate_path_cpt_value(outcomes, v)

            # Manual calculation with objective probabilities
            expected = sum(o.probability * v(o.reward) for o in outcomes)

            assert abs(cpt - expected) < 1e-10, \
                f"Row {row}: CPT uses probability weighting when it shouldn't"

    def test_value_function_applied_correctly(self):
        """Verify v() is applied to rewards, not to probabilities."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.0,  # Deterministic
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)
        v = CPTValueFunction()

        for row, outcomes in distributions.items():
            assert len(outcomes) == 1  # Deterministic
            reward = outcomes[0].reward
            probability = outcomes[0].probability

            cpt = calculate_path_cpt_value(outcomes, v)

            # Should be: 1.0 * v(reward)
            expected = probability * v(reward)
            assert abs(cpt - expected) < 1e-10


class TestReferencePointSensitivity:
    """Tests for reference point sensitivity in CPT."""

    def test_default_reference_point_zero(self):
        """Default reference point should be 0."""
        v = CPTValueFunction()
        assert v.reference_point == 0.0

    def test_reference_point_shifts_gains_losses(self):
        """Reference point should determine what counts as gain vs loss."""
        # With reference=0, all negative rewards are losses
        v0 = CPTValueFunction(reference_point=0.0)
        assert v0(-5) < 0  # Loss
        assert v0(5) > 0   # Gain

        # With reference=-10, reward=-5 is a gain!
        v10 = CPTValueFunction(reference_point=-10.0)
        assert v10(-5) > 0  # Gain (better than -10)
        assert v10(-15) < 0  # Loss (worse than -10)

    def test_reference_point_affects_cpt_values(self):
        """Different reference points should produce different CPT values."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        v_ref0 = CPTValueFunction(reference_point=0.0)
        v_ref_neg10 = CPTValueFunction(reference_point=-10.0)

        for row, outcomes in distributions.items():
            cpt0 = calculate_path_cpt_value(outcomes, v_ref0)
            cpt_neg10 = calculate_path_cpt_value(outcomes, v_ref_neg10)

            # Different reference points should give different values
            assert cpt0 != cpt_neg10, \
                f"Row {row}: same CPT value with different reference points"

    def test_all_negative_rewards_all_losses(self):
        """With reference=0, all negative rewards are losses."""
        v = CPTValueFunction(reference_point=0.0)
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        for row, outcomes in distributions.items():
            for o in outcomes:
                # All rewards are negative
                assert o.reward < 0
                # So all v(reward) should be negative (losses)
                assert v(o.reward) < 0


class TestSoftmaxTemperature:
    """Tests for softmax behavior (implicit temperature=1)."""

    def test_softmax_temperature_one(self):
        """Current implementation uses temperature=1 (no scaling)."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        result = compare_value_frameworks(config, verbose=False)

        # Verify softmax is applied directly (temperature=1, with numerical stability)
        ev_values = np.array(list(result['ev_values'].values()))
        expected_probs = np.exp(ev_values - ev_values.max()) / np.exp(ev_values - ev_values.max()).sum()
        actual_probs = np.array(list(result['ev_probabilities'].values()))

        np.testing.assert_array_almost_equal(expected_probs, actual_probs)

    def test_softmax_with_temperature_scaling(self):
        """Demonstrate effect of temperature scaling (not in current implementation)."""
        values = np.array([-8.0, -10.0, -12.0, -6.0])

        # Temperature = 1 (current)
        probs_t1 = np.exp(values) / np.exp(values).sum()

        # Temperature = 0.5 (sharper distribution)
        probs_t05 = np.exp(values / 0.5) / np.exp(values / 0.5).sum()

        # Temperature = 2 (flatter distribution)
        probs_t2 = np.exp(values / 2) / np.exp(values / 2).sum()

        # Lower temperature = more concentrated on max
        assert probs_t05.max() > probs_t1.max() > probs_t2.max()


class TestLossAversionCoefficient:
    """Tests for loss aversion coefficient (λ) sensitivity."""

    @pytest.mark.parametrize("lambda_", [1.0, 1.5, 2.25, 3.0, 5.0])
    def test_lambda_scales_losses(self, lambda_):
        """Lambda should scale the magnitude of losses."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=lambda_)

        # For x < 0: v(x) = -λ * (-x)^β
        x = -10
        expected = -lambda_ * (10 ** 0.88)
        assert abs(v(x) - expected) < 1e-10

    def test_higher_lambda_more_loss_averse(self):
        """Higher lambda should make losses relatively worse."""
        lambdas = [1.0, 2.25, 5.0]
        loss_values = []

        for lambda_ in lambdas:
            v = CPTValueFunction(lambda_=lambda_)
            loss_values.append(v(-10))

        # Higher lambda = more negative loss value
        for i in range(len(lambdas) - 1):
            assert loss_values[i] > loss_values[i + 1], \
                f"λ={lambdas[i]} should give less negative loss than λ={lambdas[i+1]}"

    def test_lambda_affects_risk_preference(self):
        """Higher lambda should increase preference for safer paths."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.3,
            'reward_step': -1.0,
            'reward_cliff': -100.0,
        }
        distributions = build_path_outcome_distributions(config)

        # Compare preferences with different lambda values
        preferences = {}
        for lambda_ in [1.0, 2.25, 5.0]:
            v = CPTValueFunction(lambda_=lambda_)
            cpt_values = {row: calculate_path_cpt_value(outcomes, v)
                         for row, outcomes in distributions.items()}
            preferred = max(cpt_values, key=cpt_values.get)
            preferences[lambda_] = preferred

        # Higher lambda should prefer safer (lower row number) or same path
        assert preferences[5.0] <= preferences[1.0], \
            f"Higher λ should prefer safer path: λ=1→row {preferences[1.0]}, λ=5→row {preferences[5.0]}"


class TestDiminishingSensitivity:
    """Tests for diminishing sensitivity (concavity/convexity)."""

    def test_gains_concave(self):
        """Gain function should be concave (diminishing marginal utility)."""
        v = CPTValueFunction(alpha=0.88)

        # Second derivative should be negative for x > 0
        # Approximate with finite differences
        x = 10
        h = 0.001
        d2v = (v(x + h) - 2 * v(x) + v(x - h)) / (h ** 2)

        assert d2v < 0, "Gain function should be concave"

    def test_losses_convex(self):
        """Loss function should be convex (diminishing marginal pain)."""
        v = CPTValueFunction(beta=0.88, lambda_=2.25)

        # For losses, the function -λ(-x)^β with 0 < β < 1
        # has increasing marginal pain as |x| increases from 0
        # But the rate of increase diminishes
        x = -10
        h = 0.001

        # Check that |v(-10)| < 10 * |v(-1)| (diminishing sensitivity)
        v_1 = abs(v(-1))
        v_10 = abs(v(-10))

        assert v_10 < 10 * v_1, "Loss function should show diminishing sensitivity"

    def test_s_shaped_curve(self):
        """CPT value function should be S-shaped (concave gains, convex losses)."""
        v = CPTValueFunction()

        # Sample points
        x_range = np.linspace(-100, 100, 201)
        values = [v(x) for x in x_range]

        # Check inflection at reference point
        zero_idx = 100  # x = 0
        assert values[zero_idx] == 0.0

        # Gains should be positive and concave
        gain_values = values[101:]
        for i in range(len(gain_values) - 2):
            # Concave: second differences should be negative
            second_diff = gain_values[i + 2] - 2 * gain_values[i + 1] + gain_values[i]
            assert second_diff < 0.01  # Allow small numerical error

        # Losses should be negative
        loss_values = values[:100]
        assert all(v < 0 for v in loss_values)


class TestStepsBeforeFallApproximation:
    """Tests for the steps_before_fall approximation."""

    def test_steps_before_fall_formula(self):
        """Examine the steps_before_fall = (nrows-1-row) + (ncols-1)/2 approximation."""
        nrows, ncols = 5, 5

        for row in range(nrows - 1):
            # Formula from code
            steps_before_fall = (nrows - 1 - row) + (ncols - 1) / 2

            # This assumes:
            # 1. UP phase completes: (nrows-1-row) steps
            # 2. Fall occurs on average halfway through RIGHT phase: (ncols-1)/2 steps

            up_steps = nrows - 1 - row
            avg_right_steps = (ncols - 1) / 2

            expected = up_steps + avg_right_steps
            assert abs(steps_before_fall - expected) < 1e-10, \
                f"Row {row}: formula mismatch"

    def test_steps_before_fall_is_approximation(self):
        """The /2 factor is an approximation for expected fall position."""
        # The actual expected fall position depends on wind_prob and distance
        # For distance=1, expected fall step can be calculated exactly

        # With d=1, fall occurs at first wind event
        # Expected position of first success in geometric distribution: 1/p
        # But bounded by h steps

        # The /2 approximation is simple average, not exact expectation
        # This test documents that it's an approximation
        nrows, ncols = 5, 5

        # Document the approximation
        for row in range(nrows - 1):
            approx = (nrows - 1 - row) + (ncols - 1) / 2
            min_steps = (nrows - 1 - row) + 1  # Fall on first RIGHT step
            max_steps = (nrows - 1 - row) + (ncols - 1)  # Fall on last RIGHT step

            # Approximation should be between min and max
            assert min_steps <= approx <= max_steps, \
                f"Row {row}: approximation {approx} not in [{min_steps}, {max_steps}]"


class TestUnionBoundApproximation:
    """Tests for the union bound approximation in cliff probability."""

    def test_union_bound_is_upper_bound(self):
        """Union bound (h-d+1)*p^d should be an upper bound on true probability."""
        # For distance d >= 2, we use union bound
        # True probability is P(at least one sequence of d consecutive winds)
        # Union bound counts overlapping sequences multiple times

        nrows, ncols = 5, 10  # More horizontal steps for clearer effect
        wind_prob = 0.3
        h = ncols - 1

        for row in range(nrows - 2):  # Rows with d >= 2
            d = nrows - 1 - row
            if d >= 2 and d <= h:
                from path_likelihood import cliff_fall_probability
                union_bound = cliff_fall_probability(row, nrows, ncols, wind_prob)

                # Monte Carlo estimate of true probability
                n_samples = 50000
                rng = np.random.default_rng(42)
                falls = 0

                for _ in range(n_samples):
                    winds = rng.random(h) < wind_prob
                    # Check for d consecutive winds
                    consecutive = 0
                    fell = False
                    for w in winds:
                        if w:
                            consecutive += 1
                            if consecutive >= d:
                                fell = True
                                break
                        else:
                            consecutive = 0
                    if fell:
                        falls += 1

                mc_prob = falls / n_samples

                # Union bound should be >= MC estimate (within statistical error)
                assert mc_prob <= union_bound + 0.02, \
                    f"Row {row} (d={d}): MC={mc_prob:.4f} > union_bound={union_bound:.4f}"
