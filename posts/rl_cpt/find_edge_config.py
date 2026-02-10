"""Grid search to find configurations where both EV and CPT prefer the risky path."""

from itertools import product

from path_likelihood import (
    cliff_fall_probability,
    build_path_outcome_distributions,
    calculate_path_expected_value,
    calculate_path_cpt_value,
)
from utils import CPTValueFunction


def evaluate_config(config: dict) -> dict:
    """Evaluate a configuration and return metrics.

    Returns dict with:
        - ev_values: dict[row, float]
        - cpt_values: dict[row, float]
        - ev_preferred: int (row)
        - cpt_preferred: int (row)
        - ev_margin: float (relative margin of preferred over safe)
        - cpt_margin: float (relative margin of preferred over safe)
    """
    distributions = build_path_outcome_distributions(config)
    value_func = CPTValueFunction()

    ev_values = {}
    cpt_values = {}

    for row, outcomes in distributions.items():
        ev_values[row] = calculate_path_expected_value(outcomes)
        cpt_values[row] = calculate_path_cpt_value(outcomes, value_func)

    ev_preferred = max(ev_values, key=ev_values.get)
    cpt_preferred = max(cpt_values, key=cpt_values.get)

    # Calculate relative margins (how much better risky is vs safe)
    # Safe row is row 0
    safe_row = 0
    ev_margin = (ev_values[ev_preferred] - ev_values[safe_row]) / abs(ev_values[safe_row]) if ev_values[safe_row] != 0 else 0
    cpt_margin = (cpt_values[cpt_preferred] - cpt_values[safe_row]) / abs(cpt_values[safe_row]) if cpt_values[safe_row] != 0 else 0

    nrows, ncols = config['shape']
    p_cliff = {row: cliff_fall_probability(row, nrows, ncols, config['wind_prob'])
               for row in distributions.keys()}

    return {
        'config': config,
        'ev_values': ev_values,
        'cpt_values': cpt_values,
        'ev_preferred': ev_preferred,
        'cpt_preferred': cpt_preferred,
        'ev_margin': ev_margin,
        'cpt_margin': cpt_margin,
        'p_cliff': p_cliff,
    }


def grid_search(target_row: int = None):
    """Search parameter space for configurations where both prefer risky path.

    Args:
        target_row: If specified, only return configs where both prefer this row.
                   If None, return any row > 0.
    """
    # Expanded search space
    wind_probs = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    cliff_rewards = [-70, -80, -85, -90, -100, -110, -120]
    step_rewards = [-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]

    # Fixed params
    shape = [5, 5]

    candidates = []

    for wind_prob, reward_cliff, reward_step in product(wind_probs, cliff_rewards, step_rewards):
        config = {
            'shape': shape,
            'wind_prob': wind_prob,
            'reward_cliff': reward_cliff,
            'reward_step': reward_step,
        }

        result = evaluate_config(config)

        # Selection criteria:
        # 1. Both prefer a risky row (row > 0), or specific target row
        # 2. Relative margins < 20% (on the verge)
        if target_row is not None:
            both_prefer_risky = result['ev_preferred'] == target_row and result['cpt_preferred'] == target_row
        else:
            both_prefer_risky = result['ev_preferred'] > 0 and result['cpt_preferred'] > 0

        on_the_verge = abs(result['ev_margin']) < 0.20 and abs(result['cpt_margin']) < 0.20

        if both_prefer_risky:
            result['both_prefer_risky'] = True
            result['on_the_verge'] = on_the_verge
            candidates.append(result)

    return candidates


def print_result(result: dict):
    """Pretty print a result."""
    config = result['config']
    print(f"\nConfig: wind={config['wind_prob']:.0%}, cliff={config['reward_cliff']}, step={config['reward_step']}")
    print(f"  EV preferred:  row {result['ev_preferred']} (margin: {result['ev_margin']:.1%})")
    print(f"  CPT preferred: row {result['cpt_preferred']} (margin: {result['cpt_margin']:.1%})")
    print(f"  On the verge:  {result['on_the_verge']}")

    nrows = config['shape'][0]
    print(f"  Row details:")
    for row in range(nrows - 1):
        p_cliff = result['p_cliff'][row]
        ev = result['ev_values'][row]
        cpt = result['cpt_values'][row]
        print(f"    Row {row}: P(cliff)={p_cliff:.3f}, EV={ev:.2f}, CPT={cpt:.2f}")


def main():
    print("=" * 60)
    print("Grid Search: Finding Configurations for Risky Path Convergence")
    print("=" * 60)

    # First search for row 2 specifically (moderately risky, per plan)
    print("\nSearching for configs where both EV and CPT prefer ROW 2...")
    candidates = grid_search(target_row=2)

    if not candidates:
        print("\nNo configs found for row 2. Trying row 1 (slightly risky)...")
        candidates = grid_search(target_row=1)

    if not candidates:
        print("\nNo configs found for row 1. Searching for any risky row...")
        candidates = grid_search(target_row=None)

    if not candidates:
        print("\nNo configurations found where both prefer risky path!")
        return

    print(f"\nFound {len(candidates)} configurations where both prefer risky path")

    # Sort by combined margin (prefer on the verge)
    candidates.sort(key=lambda r: abs(r['ev_margin']) + abs(r['cpt_margin']))

    # Show top candidates
    print("\n" + "=" * 60)
    print("Top 5 Candidates (sorted by margin - closest to the verge first)")
    print("=" * 60)

    for result in candidates[:5]:
        print_result(result)

    # Find best "on the verge" candidate
    on_verge = [c for c in candidates if c['on_the_verge']]

    print("\n" + "=" * 60)
    if on_verge:
        print(f"BEST 'ON THE VERGE' CONFIGURATION ({len(on_verge)} found)")
        print("=" * 60)
        best = on_verge[0]
        print_result(best)

        # Print YAML config
        print("\nRecommended config.yaml:")
        print("```yaml")
        print(f"shape: {best['config']['shape']}")
        print(f"stochasticity: windy")
        print(f"reward_cliff: {best['config']['reward_cliff']}")
        print(f"reward_step: {best['config']['reward_step']}")
        print(f"wind_prob: {best['config']['wind_prob']}")
        print("timesteps: 300000")
        print("n_eval_episodes: 4")
        print("```")
    else:
        print("NO 'ON THE VERGE' CONFIGURATIONS FOUND")
        print("=" * 60)
        print("\nBest available (smallest margins):")
        best = candidates[0]
        print_result(best)

        print("\nRecommended config.yaml:")
        print("```yaml")
        print(f"shape: {best['config']['shape']}")
        print(f"stochasticity: windy")
        print(f"reward_cliff: {best['config']['reward_cliff']}")
        print(f"reward_step: {best['config']['reward_step']}")
        print(f"wind_prob: {best['config']['wind_prob']}")
        print("timesteps: 300000")
        print("n_eval_episodes: 4")
        print("```")

    return candidates[0] if candidates else None


if __name__ == "__main__":
    main()
