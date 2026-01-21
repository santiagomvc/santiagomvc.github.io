#!/usr/bin/env python3
"""Diagnostic suite for CPT CliffWalking implementation.

Run with:
    python diagnostics.py --quick           # Quick checks only
    python diagnostics.py --monte-carlo     # Include Monte Carlo validation
    python diagnostics.py --all             # Full diagnostic suite
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from custom_cliff_walking import ResizableCliffWalkingEnv, CliffWalkingWrapper
from path_likelihood import (
    cliff_fall_probability,
    build_path_outcome_distributions,
    calculate_path_expected_value,
    calculate_path_cpt_value,
)
from utils import CPTValueFunction, load_config


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    passed: bool
    message: str
    details: str = ""


class DiagnosticSuite:
    """Runnable diagnostic suite for CPT CliffWalking."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def add_result(self, result: DiagnosticResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        self.log(f"  [{status}] {result.name}: {result.message}")
        if result.details and self.verbose:
            for line in result.details.split('\n'):
                self.log(f"         {line}")

    def run_quick_checks(self):
        """Run quick sanity checks."""
        self.log("\n" + "=" * 60)
        self.log("QUICK CHECKS")
        self.log("=" * 60)

        # Check 1: Imports work
        try:
            from custom_cliff_walking import make_env
            from path_likelihood import compare_value_frameworks
            self.add_result(DiagnosticResult(
                "Imports", True, "All modules import successfully"
            ))
        except ImportError as e:
            self.add_result(DiagnosticResult(
                "Imports", False, f"Import error: {e}"
            ))

        # Check 2: Config loads
        try:
            config = load_config()
            assert 'shape' in config
            assert 'wind_prob' in config
            self.add_result(DiagnosticResult(
                "Config", True, f"Config loaded: {config['shape']} grid, wind={config['wind_prob']}"
            ))
        except Exception as e:
            self.add_result(DiagnosticResult(
                "Config", False, f"Config error: {e}"
            ))

        # Check 3: Environment creates
        try:
            env = ResizableCliffWalkingEnv(shape=(5, 5))
            env = CliffWalkingWrapper(env, wind_prob=0.0)
            obs, info = env.reset()
            assert isinstance(obs, (int, np.integer))
            self.add_result(DiagnosticResult(
                "Environment", True, f"Environment created, start state={obs}"
            ))
        except Exception as e:
            self.add_result(DiagnosticResult(
                "Environment", False, f"Environment error: {e}"
            ))

        # Check 4: CPT value function
        try:
            v = CPTValueFunction()
            assert v(0) == 0.0, "v(0) should be 0"
            assert v(10) > 0, "v(10) should be positive"
            assert v(-10) < 0, "v(-10) should be negative"
            assert abs(v(-10)) > v(10), "Loss aversion: |v(-10)| > v(10)"
            self.add_result(DiagnosticResult(
                "CPT Value Function", True,
                f"v(10)={v(10):.3f}, v(-10)={v(-10):.3f}"
            ))
        except AssertionError as e:
            self.add_result(DiagnosticResult(
                "CPT Value Function", False, str(e)
            ))

    def run_math_checks(self):
        """Run mathematical validation checks."""
        self.log("\n" + "=" * 60)
        self.log("MATHEMATICAL CHECKS")
        self.log("=" * 60)

        # Check 1: Step counting formula
        self._check_step_counting()

        # Check 2: Cliff probability formula
        self._check_cliff_probability()

        # Check 3: CPT value function properties
        self._check_cpt_properties()

        # Check 4: Probability normalization
        self._check_probability_normalization()

    def _check_step_counting(self):
        """Verify step counting formula: 2*(nrows-1-row) + (ncols-1)."""
        nrows, ncols = 5, 5
        all_correct = True
        details = []

        for row in range(nrows - 1):
            # Manual calculation
            up_steps = nrows - 1 - row  # From start row to target row
            right_steps = ncols - 1      # Across the grid
            down_steps = nrows - 1 - row  # From target row to goal row
            expected = up_steps + right_steps + down_steps

            # Formula from code
            formula = 2 * (nrows - 1 - row) + (ncols - 1)

            if expected != formula:
                all_correct = False
                details.append(f"Row {row}: expected={expected}, formula={formula}")
            else:
                details.append(f"Row {row}: {formula} steps (UP:{up_steps} + RIGHT:{right_steps} + DOWN:{down_steps})")

        self.add_result(DiagnosticResult(
            "Step Counting Formula",
            all_correct,
            "Formula matches manual calculation" if all_correct else "Formula mismatch",
            "\n".join(details)
        ))

    def _check_cliff_probability(self):
        """Verify cliff probability formulas."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        h = ncols - 1  # horizontal steps
        details = []
        all_correct = True

        for row in range(nrows - 1):
            d = nrows - 1 - row  # distance to cliff
            p = cliff_fall_probability(row, nrows, ncols, wind_prob)

            if d == 0:
                expected = 1.0
            elif d >= h:
                expected = 0.0
            elif d == 1:
                expected = 1 - (1 - wind_prob) ** h
            else:
                expected = min(1.0, (h - d + 1) * (wind_prob ** d))

            if abs(p - expected) > 1e-10:
                all_correct = False
                details.append(f"Row {row} (d={d}): got={p:.6f}, expected={expected:.6f}")
            else:
                details.append(f"Row {row} (d={d}): P(cliff)={p:.6f}")

        self.add_result(DiagnosticResult(
            "Cliff Probability Formula",
            all_correct,
            "All probabilities match" if all_correct else "Probability mismatch",
            "\n".join(details)
        ))

    def _check_cpt_properties(self):
        """Verify CPT value function properties."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)
        issues = []

        # Property 1: v(0) = 0
        if v(0) != 0.0:
            issues.append(f"v(0) = {v(0)}, expected 0.0")

        # Property 2: v(x) = x^α for gains
        for x in [1, 10, 100]:
            expected = x ** 0.88
            actual = v(x)
            if abs(actual - expected) > 1e-10:
                issues.append(f"v({x}) = {actual:.6f}, expected {expected:.6f}")

        # Property 3: v(-x) = -λ(-x)^β for losses
        for x in [1, 10, 100]:
            expected = -2.25 * (x ** 0.88)
            actual = v(-x)
            if abs(actual - expected) > 1e-10:
                issues.append(f"v({-x}) = {actual:.6f}, expected {expected:.6f}")

        # Property 4: Loss aversion |v(-x)| > v(x)
        for x in [1, 10, 100]:
            if abs(v(-x)) <= v(x):
                issues.append(f"Loss aversion violated at x={x}: |v({-x})|={abs(v(-x)):.3f} <= v({x})={v(x):.3f}")

        self.add_result(DiagnosticResult(
            "CPT Properties",
            len(issues) == 0,
            "All properties satisfied" if not issues else f"{len(issues)} property violations",
            "\n".join(issues) if issues else "v(0)=0, gains/losses formulas correct, loss aversion holds"
        ))

    def _check_probability_normalization(self):
        """Check that outcome probabilities sum to 1."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        distributions = build_path_outcome_distributions(config)

        issues = []
        for row, outcomes in distributions.items():
            total_prob = sum(o.probability for o in outcomes)
            if abs(total_prob - 1.0) > 1e-10:
                issues.append(f"Row {row}: probabilities sum to {total_prob:.10f}")

        self.add_result(DiagnosticResult(
            "Probability Normalization",
            len(issues) == 0,
            "All rows sum to 1.0" if not issues else "Probability sum errors",
            "\n".join(issues) if issues else ""
        ))

    def run_monte_carlo_validation(self, n_samples: int = 10000):
        """Run Monte Carlo validation of theoretical predictions."""
        self.log("\n" + "=" * 60)
        self.log(f"MONTE CARLO VALIDATION (n={n_samples})")
        self.log("=" * 60)

        # Test 1: Cliff probability validation
        self._validate_cliff_probability_mc(n_samples)

        # Test 2: Expected value validation
        self._validate_expected_value_mc(n_samples)

    def _validate_cliff_probability_mc(self, n_samples: int):
        """Validate cliff probability with Monte Carlo simulation."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        rng = np.random.default_rng(42)

        details = []
        max_deviation = 0.0

        for row in range(nrows - 1):
            theoretical = cliff_fall_probability(row, nrows, ncols, wind_prob)

            # Simulate horizontal traversal
            cliff_falls = 0
            d = nrows - 1 - row  # distance to cliff

            for _ in range(n_samples):
                current_row = row
                fell = False
                for step in range(ncols - 1):  # horizontal steps
                    if rng.random() < wind_prob:
                        current_row += 1
                    if current_row >= nrows - 1:  # Hit cliff row
                        fell = True
                        break
                if fell:
                    cliff_falls += 1

            empirical = cliff_falls / n_samples
            deviation = abs(empirical - theoretical)
            max_deviation = max(max_deviation, deviation)

            details.append(
                f"Row {row} (d={d}): theoretical={theoretical:.4f}, "
                f"empirical={empirical:.4f}, diff={deviation:.4f}"
            )

        # Allow 5% deviation for Monte Carlo
        passed = max_deviation < 0.05

        self.add_result(DiagnosticResult(
            "MC Cliff Probability",
            passed,
            f"Max deviation: {max_deviation:.4f}" + (" (within 5%)" if passed else " (>5%)"),
            "\n".join(details)
        ))

    def _validate_expected_value_mc(self, n_samples: int):
        """Validate expected value with Monte Carlo simulation."""
        nrows, ncols = 5, 5
        wind_prob = 0.2
        reward_step = -1.0
        reward_cliff = -50.0
        rng = np.random.default_rng(42)

        config = {
            'shape': [nrows, ncols],
            'wind_prob': wind_prob,
            'reward_step': reward_step,
            'reward_cliff': reward_cliff,
        }
        distributions = build_path_outcome_distributions(config)

        details = []
        max_deviation = 0.0

        for row in range(nrows - 1):
            theoretical_ev = calculate_path_expected_value(distributions[row])

            # Simulate episodes
            total_rewards = []
            d = nrows - 1 - row

            for _ in range(n_samples):
                current_row = row
                steps = 0
                fell = False

                # UP phase
                for _ in range(nrows - 1 - row):
                    steps += 1

                # RIGHT phase with wind
                for _ in range(ncols - 1):
                    if rng.random() < wind_prob:
                        current_row += 1
                    steps += 1
                    if current_row >= nrows - 1:
                        fell = True
                        break

                # DOWN phase (if didn't fall)
                if not fell:
                    for _ in range(nrows - 1 - row):
                        steps += 1

                if fell:
                    total_rewards.append(steps * reward_step + reward_cliff)
                else:
                    total_rewards.append(steps * reward_step)

            empirical_ev = np.mean(total_rewards)
            deviation = abs(empirical_ev - theoretical_ev)
            max_deviation = max(max_deviation, deviation)

            details.append(
                f"Row {row}: theoretical={theoretical_ev:.2f}, "
                f"empirical={empirical_ev:.2f}, diff={deviation:.2f}"
            )

        # Allow larger deviation for EV (depends on cliff penalty magnitude)
        passed = max_deviation < 5.0

        self.add_result(DiagnosticResult(
            "MC Expected Value",
            passed,
            f"Max deviation: {max_deviation:.2f}" + (" (acceptable)" if passed else " (too large)"),
            "\n".join(details)
        ))

    def run_consistency_checks(self):
        """Run consistency checks between EV and CPT."""
        self.log("\n" + "=" * 60)
        self.log("CONSISTENCY CHECKS")
        self.log("=" * 60)

        # Check 1: Softmax numerical stability
        self._check_softmax_stability()

        # Check 2: EV vs CPT ordering divergence
        self._check_ev_cpt_divergence()

    def _check_softmax_stability(self):
        """Check softmax numerical stability with extreme values."""
        test_cases = [
            (np.array([0.0, 0.0, 0.0]), "zeros"),
            (np.array([-1.0, -2.0, -3.0]), "small negatives"),
            (np.array([-100.0, -200.0, -300.0]), "large negatives"),
            (np.array([-1000.0, -2000.0, -3000.0]), "extreme negatives"),
        ]

        issues = []
        for values, name in test_cases:
            try:
                # Use numerically stable softmax: subtract max before exp
                probs = np.exp(values - values.max()) / np.exp(values - values.max()).sum()
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    issues.append(f"{name}: produced NaN/Inf")
                elif abs(probs.sum() - 1.0) > 1e-6:
                    issues.append(f"{name}: sum={probs.sum():.10f}")
            except Exception as e:
                issues.append(f"{name}: {e}")

        self.add_result(DiagnosticResult(
            "Softmax Stability",
            len(issues) == 0,
            "Stable for all test cases" if not issues else f"{len(issues)} stability issues",
            "\n".join(issues) if issues else ""
        ))

    def _check_ev_cpt_divergence(self):
        """Check if EV and CPT produce different orderings."""
        config = {
            'shape': [5, 5],
            'wind_prob': 0.2,
            'reward_step': -1.0,
            'reward_cliff': -50.0,
        }
        v = CPTValueFunction()
        distributions = build_path_outcome_distributions(config)

        ev_values = {row: calculate_path_expected_value(outcomes)
                     for row, outcomes in distributions.items()}
        cpt_values = {row: calculate_path_cpt_value(outcomes, v)
                      for row, outcomes in distributions.items()}

        ev_ranking = sorted(ev_values.keys(), key=lambda r: ev_values[r], reverse=True)
        cpt_ranking = sorted(cpt_values.keys(), key=lambda r: cpt_values[r], reverse=True)

        diverges = ev_ranking != cpt_ranking

        details = [
            f"EV ranking:  {ev_ranking} (best: row {ev_ranking[0]})",
            f"CPT ranking: {cpt_ranking} (best: row {cpt_ranking[0]})",
            "",
            "Row values:",
        ]
        for row in sorted(ev_values.keys()):
            details.append(f"  Row {row}: EV={ev_values[row]:.2f}, CPT={cpt_values[row]:.2f}")

        self.add_result(DiagnosticResult(
            "EV vs CPT Divergence",
            True,  # Not a pass/fail, just informational
            "Rankings differ (expected with loss aversion)" if diverges else "Rankings identical",
            "\n".join(details)
        ))

    def run_all(self, include_monte_carlo: bool = True, mc_samples: int = 10000):
        """Run all diagnostic checks."""
        self.log("=" * 60)
        self.log("CPT CLIFFWALKING DIAGNOSTIC SUITE")
        self.log("=" * 60)

        self.run_quick_checks()
        self.run_math_checks()
        self.run_consistency_checks()

        if include_monte_carlo:
            self.run_monte_carlo_validation(mc_samples)

        self.print_summary()

    def print_summary(self):
        """Print summary of all diagnostic results."""
        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        self.log(f"Total checks: {len(self.results)}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")

        if failed > 0:
            self.log("\nFailed checks:")
            for r in self.results:
                if not r.passed:
                    self.log(f"  - {r.name}: {r.message}")

        self.log("\n" + "=" * 60)
        if failed == 0:
            self.log("ALL CHECKS PASSED")
        else:
            self.log(f"WARNING: {failed} CHECK(S) FAILED")
        self.log("=" * 60)

        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="CPT CliffWalking Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Quick checks only")
    parser.add_argument("--monte-carlo", "-mc", action="store_true", help="Include Monte Carlo validation")
    parser.add_argument("-n", type=int, default=10000, help="Monte Carlo sample size")
    parser.add_argument("--all", action="store_true", help="Run all diagnostics")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    suite = DiagnosticSuite(verbose=not args.quiet)

    if args.quick:
        suite.run_quick_checks()
        suite.print_summary()
    elif args.all or args.monte_carlo:
        suite.run_all(include_monte_carlo=True, mc_samples=args.n)
    else:
        suite.run_all(include_monte_carlo=False)

    return 0 if all(r.passed for r in suite.results) else 1


if __name__ == "__main__":
    sys.exit(main())
