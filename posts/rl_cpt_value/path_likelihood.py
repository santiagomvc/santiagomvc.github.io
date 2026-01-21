"""Path likelihood calculations for CliffWalking with CPT value comparison."""

from dataclasses import dataclass
from typing import List

import numpy as np

from utils import load_config, CPTValueFunction


@dataclass
class PathOutcome:
    """A possible outcome when taking a specific path."""
    reward: float
    probability: float
    is_success: bool
    description: str = ""


def cliff_fall_probability(row: int, nrows: int, ncols: int, wind_prob: float) -> float:
    """Calculate probability of falling into cliff when traversing at given row.

    Args:
        row: The row index to traverse (0 = top, nrows-2 = one above cliff)
        nrows: Total number of rows
        ncols: Total number of columns
        wind_prob: Probability of wind pushing DOWN on each step

    Returns:
        Probability of falling into the cliff
    """
    distance_to_cliff = nrows - 1 - row
    horizontal_steps = ncols - 1

    if distance_to_cliff == 0:
        return 1.0  # Row is the cliff row itself

    if wind_prob == 0:
        return 0.0  # No wind, no risk

    if distance_to_cliff >= horizontal_steps:
        return 0.0  # Can't fall even with wind on every step

    # Distance=1: any single wind = cliff
    if distance_to_cliff == 1:
        return 1 - (1 - wind_prob) ** horizontal_steps

    # Distance>=2: need consecutive winds
    d = distance_to_cliff
    h = horizontal_steps
    if d > h:
        return 0.0
    return min(1.0, (h - d + 1) * (wind_prob ** d))


def calculate_path_expected_value(outcomes: List[PathOutcome]) -> float:
    """Standard expected value: E[R] = Σ p_i * r_i"""
    return sum(o.reward * o.probability for o in outcomes)


def calculate_path_cpt_value(
    outcomes: List[PathOutcome],
    value_func: CPTValueFunction = None,
) -> float:
    """CPT value: V = Σ p_i * v(r_i)"""
    if value_func is None:
        value_func = CPTValueFunction()
    return sum(o.probability * value_func(o.reward) for o in outcomes)


def build_path_outcome_distributions(
    env_config: dict = None,
) -> dict[int, List[PathOutcome]]:
    """Build outcome distributions for each viable row path.

    Args:
        env_config: Config with 'shape', 'wind_prob', 'reward_step', 'reward_cliff'
                   Uses load_config() if None

    Returns:
        Dict mapping row index to list of PathOutcome objects
    """
    if env_config is None:
        env_config = load_config()

    nrows, ncols = env_config['shape']
    wind_prob = env_config.get('wind_prob', 0.0)
    reward_step = env_config['reward_step']
    reward_cliff = env_config['reward_cliff']

    distributions = {}

    for row in range(nrows - 1):  # Viable paths: rows 0 to nrows-2
        total_steps = 2 * (nrows - 1 - row) + (ncols - 1)
        p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
        p_success = 1 - p_cliff

        outcomes = []

        if p_success > 0:
            outcomes.append(PathOutcome(
                reward=total_steps * reward_step,
                probability=p_success,
                is_success=True,
                description=f"Success via row {row}"
            ))

        if p_cliff > 0:
            steps_before_fall = (nrows - 1 - row) + (ncols - 1) / 2
            outcomes.append(PathOutcome(
                reward=steps_before_fall * reward_step + reward_cliff,
                probability=p_cliff,
                is_success=False,
                description=f"Cliff fall from row {row}"
            ))

        distributions[row] = outcomes

    return distributions


def compare_value_frameworks(
    env_config: dict = None,
    cpt_params: dict = None,
    verbose: bool = True,
) -> dict:
    """Compare path values between EV and CPT frameworks.

    Args:
        env_config: Environment config (uses load_config() if None)
        cpt_params: CPT parameters {"alpha", "beta", "lambda_"}
        verbose: Print detailed comparison

    Returns:
        Dict with ev_values, cpt_values, probabilities, and preferred paths
    """
    if env_config is None:
        env_config = load_config()

    params = cpt_params or {}
    value_func = CPTValueFunction(
        alpha=params.get("alpha", 0.88),
        beta=params.get("beta", 0.88),
        lambda_=params.get("lambda_", 2.25),
    )

    distributions = build_path_outcome_distributions(env_config)

    ev_values = {}
    cpt_values = {}

    for row, outcomes in distributions.items():
        ev_values[row] = calculate_path_expected_value(outcomes)
        cpt_values[row] = calculate_path_cpt_value(outcomes, value_func)

    # Softmax to get choice probabilities
    ev_arr = np.array(list(ev_values.values()))
    cpt_arr = np.array(list(cpt_values.values()))

    ev_probs = np.exp(ev_arr - ev_arr.max()) / np.exp(ev_arr - ev_arr.max()).sum()
    cpt_probs = np.exp(cpt_arr - cpt_arr.max()) / np.exp(cpt_arr - cpt_arr.max()).sum()

    rows = list(ev_values.keys())
    ev_probabilities = dict(zip(rows, ev_probs))
    cpt_probabilities = dict(zip(rows, cpt_probs))

    ev_preferred = max(ev_values, key=ev_values.get)
    cpt_preferred = max(cpt_values, key=cpt_values.get)

    result = {
        'env_config': env_config,
        'path_outcomes': distributions,
        'ev_values': ev_values,
        'cpt_values': cpt_values,
        'ev_probabilities': ev_probabilities,
        'cpt_probabilities': cpt_probabilities,
        'ev_preferred_row': ev_preferred,
        'cpt_preferred_row': cpt_preferred,
    }

    if verbose:
        nrows, ncols = env_config['shape']
        print("=" * 60)
        print("Framework Comparison: Expected Value vs CPT Value Function")
        print("=" * 60)
        print(f"Grid: {nrows}x{ncols}, Wind: {env_config.get('wind_prob', 0):.0%}")
        print(f"Rewards: step={env_config['reward_step']}, cliff={env_config['reward_cliff']}")
        print()
        print(f"{'Row':<6} {'Steps':<8} {'P(cliff)':<12} {'EV':<12} {'CPT':<12}")
        print("-" * 50)
        for row in rows:
            steps = 2 * (nrows - 1 - row) + (ncols - 1)
            p_cliff = cliff_fall_probability(row, nrows, ncols, env_config.get('wind_prob', 0))
            print(f"{row:<6} {steps:<8} {p_cliff:<12.4f} {ev_values[row]:<12.2f} {cpt_values[row]:<12.2f}")
        print()
        print(f"EV preferred row:  {ev_preferred}")
        print(f"CPT preferred row: {cpt_preferred}")

    return result


if __name__ == "__main__":
    # Test with default config
    compare_value_frameworks()
