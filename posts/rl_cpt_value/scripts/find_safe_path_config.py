"""Grid search to find configurations where both EV and CPT prefer a SAFE path.

Target: row <= nrows-3 (i.e., 2+ rows above the cliff).

User constraints:
- goal_reward > 0, step_reward = 0, cliff_reward > 0 (but < goal_reward)
- wind_prob between 5% and 15%
- REINFORCE baseline: EMA, CPT baseline: min

Key insight with step_reward=0:
  G = gamma^(N-1) * terminal_reward
  Gamma is the ONLY mechanism differentiating paths by length.
  Shorter (riskier) paths get less discounting but face higher cliff probability.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from path_likelihood import cliff_fall_probability
from utils import CPTValueFunction


def path_steps(row: int, nrows: int, ncols: int) -> int:
    """Total steps for canonical path via given row: up + right + down."""
    return 2 * (nrows - 1 - row) + (ncols - 1)


def discounted_ev(
    row: int, nrows: int, ncols: int, wind_prob: float,
    goal_reward: float, cliff_reward: float, gamma: float,
) -> float:
    """Compute gamma-discounted expected value for a row path with step_reward=0.

    G_success = gamma^(steps-1) * goal_reward
    G_cliff   = gamma^(steps_cliff-1) * cliff_reward
    EV = p_success * G_success + p_cliff * G_cliff
    """
    steps = path_steps(row, nrows, ncols)
    p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
    p_success = 1 - p_cliff

    g_success = (gamma ** (steps - 1)) * goal_reward
    # Approximate cliff steps: go up + half of horizontal traverse
    steps_cliff = (nrows - 1 - row) + (ncols - 1) // 2
    steps_cliff = max(steps_cliff, 1)
    g_cliff = (gamma ** (steps_cliff - 1)) * cliff_reward

    return p_success * g_success + p_cliff * g_cliff


def discounted_cpt(
    row: int, nrows: int, ncols: int, wind_prob: float,
    goal_reward: float, cliff_reward: float, gamma: float,
    value_func: CPTValueFunction = None,
) -> float:
    """Compute gamma-discounted CPT value for a row path with step_reward=0.

    Same structure as EV but applies CPT value function to outcomes.
    Since all rewards are positive, v(x) = x^alpha.
    """
    if value_func is None:
        value_func = CPTValueFunction()

    steps = path_steps(row, nrows, ncols)
    p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
    p_success = 1 - p_cliff

    g_success = (gamma ** (steps - 1)) * goal_reward
    steps_cliff = (nrows - 1 - row) + (ncols - 1) // 2
    steps_cliff = max(steps_cliff, 1)
    g_cliff = (gamma ** (steps_cliff - 1)) * cliff_reward

    return p_success * value_func(g_success) + p_cliff * value_func(g_cliff)


def monte_carlo_validate(
    row: int, nrows: int, ncols: int, wind_prob: float,
    goal_reward: float, cliff_reward: float, gamma: float,
    n_simulations: int = 50000,
) -> float:
    """Monte Carlo validation of discounted EV for a row path."""
    rng = np.random.default_rng(42)
    ncols_steps = ncols - 1
    distance_to_cliff = nrows - 1 - row

    total = 0.0
    for _ in range(n_simulations):
        # Simulate: go up to row, traverse right, go down to goal
        up_steps = nrows - 1 - row
        current_vertical = 0  # distance below row 0 (starts at row, so offset = row after going up)

        # Agent goes up first (no wind during up phase, deterministic)
        # Then traverses right at row 0
        wind_pushes = 0
        fell = False
        step_count = up_steps  # up phase steps

        for h in range(ncols_steps):
            step_count += 1
            if rng.random() < wind_prob:
                wind_pushes += 1
            if wind_pushes >= distance_to_cliff:
                # Actually: agent is at row 0, wind pushes down
                # distance_to_cliff from row 0 is nrows-1
                pass

        # Recalculate: from row 0, need nrows-1 pushes to reach cliff
        # That's actually very unlikely for large grids
        # Let's use the canonical model: traverse at the given row directly
        # Wind pushes down, need distance_to_cliff pushes to hit cliff
        wind_pushes = 0
        fell = False
        step_count = 0

        # Phase: traverse right at given row
        for h in range(ncols_steps):
            step_count += 1
            if rng.random() < wind_prob:
                wind_pushes += 1
                if wind_pushes >= distance_to_cliff:
                    fell = True
                    break
            else:
                wind_pushes = 0  # Non-consecutive resets (simplified model)

        if fell:
            g = (gamma ** (step_count - 1)) * cliff_reward
        else:
            # Complete the path: up + right + down
            total_steps = path_steps(row, nrows, ncols)
            g = (gamma ** (total_steps - 1)) * goal_reward

        total += g

    return total / n_simulations


def evaluate_config(
    shape: list, wind_prob: float, goal_reward: float,
    cliff_reward: float, gamma: float,
    cpt_params: dict = None,
) -> dict:
    """Evaluate a configuration and return metrics."""
    nrows, ncols = shape
    params = cpt_params or {}
    value_func = CPTValueFunction(
        alpha=params.get("alpha", 0.88),
        beta=params.get("beta", 0.88),
        lambda_=params.get("lambda_", 2.25),
    )

    ev_values = {}
    cpt_values = {}

    for row in range(nrows - 1):
        ev_values[row] = discounted_ev(
            row, nrows, ncols, wind_prob, goal_reward, cliff_reward, gamma
        )
        cpt_values[row] = discounted_cpt(
            row, nrows, ncols, wind_prob, goal_reward, cliff_reward, gamma, value_func
        )

    ev_preferred = max(ev_values, key=ev_values.get)
    cpt_preferred = max(cpt_values, key=cpt_values.get)

    # Safe = 2+ rows above cliff
    safe_threshold = nrows - 3
    ev_is_safe = ev_preferred <= safe_threshold
    cpt_is_safe = cpt_preferred <= safe_threshold

    # CRITICAL: Check that the safe path EV beats an immediate cliff fall.
    # Immediate cliff: agent steps right from start (nrows-1, 0) into cliff at (nrows-1, 1).
    # Return = cliff_reward (1 step, gamma^0 * cliff_reward = cliff_reward).
    immediate_cliff_return = cliff_reward
    goal_beats_cliff = ev_values.get(ev_preferred, 0) > immediate_cliff_return

    # Compute EV gap between safe winner and best risky row
    risky_row = nrows - 2  # 1 above cliff
    if ev_is_safe:
        ev_gap = (ev_values[ev_preferred] - ev_values[risky_row]) / abs(ev_values[ev_preferred]) if ev_values[ev_preferred] != 0 else 0
    else:
        ev_gap = 0

    # Also compute the margin over immediate cliff (how much better goal is)
    cliff_margin = (ev_values.get(ev_preferred, 0) - immediate_cliff_return) / abs(ev_values.get(ev_preferred, 1)) if ev_values.get(ev_preferred, 0) != 0 else 0

    return {
        'shape': shape,
        'wind_prob': wind_prob,
        'goal_reward': goal_reward,
        'cliff_reward': cliff_reward,
        'gamma': gamma,
        'ev_values': ev_values,
        'cpt_values': cpt_values,
        'ev_preferred': ev_preferred,
        'cpt_preferred': cpt_preferred,
        'ev_is_safe': ev_is_safe,
        'cpt_is_safe': cpt_is_safe,
        'ev_gap': ev_gap,
        'goal_beats_cliff': goal_beats_cliff,
        'cliff_margin': cliff_margin,
    }


def grid_search():
    """Search parameter space for configs where both EV and CPT prefer a safe path.

    Selection criteria:
    1. EV-preferred row is "safe" (row <= nrows-3, i.e., 2+ above cliff)
    2. CPT-preferred row is also safe
    3. EV gap between safe and risky < 5% (creates indecision)
    4. EV gap > 0 (safe must win)
    """
    shapes = [[4, 5], [4, 6], [4, 7], [4, 8], [5, 5], [5, 6], [5, 7], [5, 8]]
    goal_rewards = [50, 75, 100, 150, 200]
    cliff_rewards = [1, 2, 3, 5, 8, 10, 15, 20]
    wind_probs = [0.05, 0.08, 0.10, 0.12, 0.15]
    gammas = [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]

    candidates = []
    total = 0

    for shape, goal_reward, cliff_reward, wind_prob, gamma in product(
        shapes, goal_rewards, cliff_rewards, wind_probs, gammas
    ):
        # Constraint: cliff_reward < goal_reward
        if cliff_reward >= goal_reward:
            continue

        total += 1
        result = evaluate_config(shape, wind_prob, goal_reward, cliff_reward, gamma)

        # Selection criteria
        if not result['goal_beats_cliff']:
            continue  # Goal must beat immediate cliff fall
        if result['cliff_margin'] < 0.50:
            continue  # Goal must beat cliff by at least 50% for trainability
        if not result['ev_is_safe']:
            continue
        if not result['cpt_is_safe']:
            continue
        if result['ev_gap'] <= 0:
            continue
        if result['ev_gap'] >= 0.10:
            continue  # Too decisive, no indecision

        candidates.append(result)

    print(f"Searched {total} configurations, found {len(candidates)} candidates.")
    return candidates


def print_result(result: dict, mc_validate: bool = False):
    """Pretty print a search result."""
    nrows, ncols = result['shape']

    print(f"\nConfig: shape={result['shape']}, wind={result['wind_prob']:.0%}, "
          f"goal={result['goal_reward']}, cliff={result['cliff_reward']}, "
          f"gamma={result['gamma']}")
    print(f"  EV preferred:  row {result['ev_preferred']} "
          f"(gap vs risky: {result['ev_gap']:.3%})")
    print(f"  CPT preferred: row {result['cpt_preferred']}")
    print(f"  Goal beats cliff: {result['goal_beats_cliff']} "
          f"(margin: {result['cliff_margin']:.3%})")

    print(f"  {'Row':<6} {'Steps':<8} {'P(cliff)':<12} {'EV':<12} {'CPT':<12}")
    print(f"  {'-'*48}")
    for row in sorted(result['ev_values'].keys()):
        steps = path_steps(row, nrows, ncols)
        p_cliff = cliff_fall_probability(row, nrows, ncols, result['wind_prob'])
        ev = result['ev_values'][row]
        cpt = result['cpt_values'][row]
        safe_marker = " <-- SAFE" if row <= nrows - 3 else ""
        pref_marker = " *EV" if row == result['ev_preferred'] else ""
        pref_marker += " *CPT" if row == result['cpt_preferred'] else ""
        print(f"  {row:<6} {steps:<8} {p_cliff:<12.4f} {ev:<12.4f} {cpt:<12.4f}{safe_marker}{pref_marker}")

    if mc_validate:
        print(f"\n  Monte Carlo validation (50k sims):")
        for row in [result['ev_preferred'], nrows - 2]:
            mc_ev = monte_carlo_validate(
                row, nrows, ncols, result['wind_prob'],
                result['goal_reward'], result['cliff_reward'], result['gamma'],
            )
            print(f"    Row {row}: MC_EV={mc_ev:.4f} (analytical: {result['ev_values'][row]:.4f})")


def generate_yaml(result: dict) -> str:
    """Generate YAML config from a result."""
    lines = [
        f"shape: {result['shape']}",
        f"stochasticity: windy",
        f"reward_cliff: {result['cliff_reward']}",
        f"reward_step: 0",
        f"reward_goal: {result['goal_reward']}",
        f"wind_prob: {result['wind_prob']}",
        f"gamma: {result['gamma']}",
        f"baseline_type: ema",
        f"baseline_type_cpt: min",
        f"timesteps: 350000",
        f"n_eval_episodes: 4",
        f"lr: 0.001",
        f"batch_size: 16",
        f"entropy_coef: 1.0",
        f"entropy_coef_final: 0.01",
        f"n_seeds: 4",
    ]
    return '\n'.join(lines) + '\n'


def main():
    print("=" * 70)
    print("Grid Search: Finding Configs for SAFE Path Preference")
    print("Target: row <= nrows-3 (2+ rows above cliff)")
    print("Constraint: step_reward=0, gamma discounts longer paths")
    print("=" * 70)

    candidates = grid_search()

    if not candidates:
        print("\nNo configurations found!")
        print("Try expanding the search space or relaxing constraints.")
        return None

    # Sort by EV gap (prefer small positive gap = maximum indecision)
    candidates.sort(key=lambda r: r['ev_gap'])

    print("\n" + "=" * 70)
    print("Top 10 Candidates (sorted by EV gap - most indecisive first)")
    print("=" * 70)

    for result in candidates[:10]:
        print_result(result)

    # Pick best: smallest gap > 0 with reasonable parameters
    best = candidates[0]

    print("\n" + "=" * 70)
    print("BEST CONFIG (most indecisive, safe path wins)")
    print("=" * 70)
    print_result(best, mc_validate=True)

    yaml_str = generate_yaml(best)
    print("\n" + "-" * 40)
    print("Recommended low_sensitivity_base.yaml:")
    print("-" * 40)
    print(yaml_str)

    # Also write it out
    config_path = Path(__file__).parent.parent / "configs" / "low_sensitivity_base.yaml"
    config_path.write_text(yaml_str)
    print(f"Written to {config_path}")

    return best


if __name__ == "__main__":
    main()
