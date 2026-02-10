"""Grid search to find configurations where both EV and CPT prefer the RISKIEST path.

Target: row (nrows-2), i.e., one row above the cliff.

User constraints:
- reward_cliff <= -80
- wind_prob between 5% and 15%

Key insight: For the riskiest path to be preferred, the step savings must outweigh
the expected cliff penalty. Smaller grids (fewer columns) reduce cliff exposure.
"""

import sys
from itertools import product
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from path_likelihood import (
    cliff_fall_probability,
    build_path_outcome_distributions,
    calculate_path_expected_value,
    calculate_path_cpt_value,
)
from utils import CPTValueFunction


def evaluate_config(config: dict, cpt_params: dict = None) -> dict:
    """Evaluate a configuration and return metrics.

    Args:
        config: Environment config with shape, wind_prob, reward_cliff, reward_step
        cpt_params: Optional CPT parameters (alpha, beta, lambda_)

    Returns dict with:
        - ev_values: dict[row, float]
        - cpt_values: dict[row, float]
        - ev_preferred: int (row)
        - cpt_preferred: int (row)
        - ev_margin: float (margin of risky vs safe)
        - cpt_margin: float (margin of risky vs safe)
        - p_cliff_risky: float (cliff probability for riskiest path)
    """
    distributions = build_path_outcome_distributions(config)

    # CPT value function with optional custom parameters
    params = cpt_params or {}
    value_func = CPTValueFunction(
        alpha=params.get("alpha", 0.88),
        beta=params.get("beta", 0.88),
        lambda_=params.get("lambda_", 2.25),
    )

    ev_values = {}
    cpt_values = {}

    for row, outcomes in distributions.items():
        ev_values[row] = calculate_path_expected_value(outcomes)
        cpt_values[row] = calculate_path_cpt_value(outcomes, value_func)

    ev_preferred = max(ev_values, key=ev_values.get)
    cpt_preferred = max(cpt_values, key=cpt_values.get)

    # Calculate margins (positive = risky is better)
    nrows = config['shape'][0]
    safe_row = 0
    risky_row = nrows - 2  # One above cliff

    ev_margin = ev_values[risky_row] - ev_values[safe_row]
    cpt_margin = cpt_values[risky_row] - cpt_values[safe_row]

    p_cliff_risky = cliff_fall_probability(
        risky_row, nrows, config['shape'][1], config['wind_prob']
    )

    return {
        'config': config,
        'cpt_params': cpt_params,
        'ev_values': ev_values,
        'cpt_values': cpt_values,
        'ev_preferred': ev_preferred,
        'cpt_preferred': cpt_preferred,
        'ev_margin': ev_margin,
        'cpt_margin': cpt_margin,
        'p_cliff_risky': p_cliff_risky,
    }


def grid_search():
    """Search parameter space for configs where both EV and CPT prefer the riskiest path.

    Returns list of candidate configurations sorted by combined margin.
    """
    # Search space per the plan
    shapes = [[4, 4], [4, 5], [4, 6], [4, 7], [4, 8]]
    wind_probs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    cliff_rewards = [-80, -90, -100, -110, -120]
    step_rewards = [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
    lambdas = [2.25, 2.0, 1.5, 1.0]

    candidates = []

    for shape, wind_prob, reward_cliff, reward_step, lambda_ in product(
        shapes, wind_probs, cliff_rewards, step_rewards, lambdas
    ):
        config = {
            'shape': shape,
            'wind_prob': wind_prob,
            'reward_cliff': reward_cliff,
            'reward_step': reward_step,
        }

        cpt_params = {'lambda_': lambda_}

        result = evaluate_config(config, cpt_params)

        nrows = shape[0]
        risky_row = nrows - 2

        # Target: both EV and CPT prefer the riskiest path
        both_prefer_risky = (
            result['ev_preferred'] == risky_row and
            result['cpt_preferred'] == risky_row
        )

        if both_prefer_risky:
            candidates.append(result)

    return candidates


def print_result(result: dict):
    """Pretty print a search result."""
    config = result['config']
    cpt_params = result.get('cpt_params') or {}
    nrows, ncols = config['shape']
    risky_row = nrows - 2

    print(f"\nConfig: shape={config['shape']}, wind={config['wind_prob']:.0%}, "
          f"cliff={config['reward_cliff']}, step={config['reward_step']}")
    if cpt_params:
        print(f"  CPT params: lambda={cpt_params.get('lambda_', 2.25)}")

    print(f"  EV preferred:  row {result['ev_preferred']} (margin vs safe: {result['ev_margin']:+.2f})")
    print(f"  CPT preferred: row {result['cpt_preferred']} (margin vs safe: {result['cpt_margin']:+.2f})")
    print(f"  P(cliff) for row {risky_row}: {result['p_cliff_risky']:.3f}")

    print(f"  Row values:")
    for row in sorted(result['ev_values'].keys()):
        ev = result['ev_values'][row]
        cpt = result['cpt_values'][row]
        p_cliff = cliff_fall_probability(row, nrows, ncols, config['wind_prob'])
        marker = " <-- TARGET" if row == risky_row else ""
        print(f"    Row {row}: EV={ev:8.2f}, CPT={cpt:8.2f}, P(cliff)={p_cliff:.3f}{marker}")


def generate_yaml_config(result: dict) -> str:
    """Generate YAML config string from a result."""
    config = result['config']
    cpt_params = result.get('cpt_params') or {}

    lines = [
        f"shape: {config['shape']}",
        f"stochasticity: windy",
        f"reward_cliff: {config['reward_cliff']}",
        f"reward_step: {config['reward_step']}",
        f"wind_prob: {config['wind_prob']}",
        f"timesteps: 300000",
        f"n_eval_episodes: 4",
        f"lr: 1e-4",
    ]

    # Add CPT params if non-default
    if cpt_params.get('lambda_') and cpt_params['lambda_'] != 2.25:
        lines.append(f"cpt_lambda: {cpt_params['lambda_']}")

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("Grid Search: Finding Configs for RISKIEST Path Preference")
    print("Target: row (nrows-2) - one row above the cliff")
    print("=" * 70)

    candidates = grid_search()

    if not candidates:
        print("\nNo configurations found where both EV and CPT prefer the riskiest path!")
        print("\nTry expanding the search space or relaxing constraints.")
        return None

    print(f"\nFound {len(candidates)} configurations!")

    # Sort by combined margin (prefer largest positive margins for robustness)
    candidates.sort(key=lambda r: r['ev_margin'] + r['cpt_margin'], reverse=True)

    # Show top candidates
    print("\n" + "=" * 70)
    print("Top 5 Candidates (sorted by margin - strongest preference first)")
    print("=" * 70)

    for result in candidates[:5]:
        print_result(result)

    # Find best candidate prioritizing:
    # 1. Standard CPT params (lambda=2.25)
    # 2. Largest margins
    standard_lambda = [c for c in candidates if c.get('cpt_params', {}).get('lambda_', 2.25) == 2.25]

    if standard_lambda:
        best = standard_lambda[0]
        print("\n" + "=" * 70)
        print("BEST CONFIG (with standard CPT lambda=2.25)")
        print("=" * 70)
    else:
        best = candidates[0]
        print("\n" + "=" * 70)
        print("BEST CONFIG (requires modified CPT lambda)")
        print("=" * 70)

    print_result(best)

    yaml_config = generate_yaml_config(best)
    print("\n" + "-" * 40)
    print("Recommended risky_path.yaml:")
    print("-" * 40)
    print(yaml_config)

    return best


if __name__ == "__main__":
    main()
