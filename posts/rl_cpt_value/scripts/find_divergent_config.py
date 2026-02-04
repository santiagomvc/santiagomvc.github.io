"""Grid search to find configs where EV prefers RISKY but CPT prefers SAFE.

Goal: find a pair of configs (baseline + divergent) sharing all params except
goal_reward. At high goal_reward both EV and CPT prefer safe; at reduced
goal_reward EV flips to risky while CPT stays safe -- demonstrating CPT's
diminishing sensitivity.

Theory (step_reward=0):
  G_success = gamma^(steps-1) * goal_reward
  G_cliff   = gamma^(steps_cliff-1) * cliff_reward
  v(x) = x^0.88 compresses the goal component slower than EV as goal decreases,
  while the cliff component is fixed. This creates a window where EV prefers
  risky but CPT still prefers safe.
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
    """Compute gamma-discounted expected value for a row path with step_reward=0."""
    steps = path_steps(row, nrows, ncols)
    p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
    p_success = 1 - p_cliff

    g_success = (gamma ** (steps - 1)) * goal_reward
    steps_cliff = (nrows - 1 - row) + (ncols - 1) // 2
    steps_cliff = max(steps_cliff, 1)
    g_cliff = (gamma ** (steps_cliff - 1)) * cliff_reward

    return p_success * g_success + p_cliff * g_cliff


def discounted_cpt(
    row: int, nrows: int, ncols: int, wind_prob: float,
    goal_reward: float, cliff_reward: float, gamma: float,
    value_func: CPTValueFunction = None,
) -> float:
    """Compute gamma-discounted CPT value for a row path with step_reward=0."""
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
        wind_pushes = 0
        fell = False
        step_count = 0

        for h in range(ncols_steps):
            step_count += 1
            if rng.random() < wind_prob:
                wind_pushes += 1
                if wind_pushes >= distance_to_cliff:
                    fell = True
                    break
            else:
                wind_pushes = 0

        if fell:
            g = (gamma ** (step_count - 1)) * cliff_reward
        else:
            total_steps = path_steps(row, nrows, ncols)
            g = (gamma ** (total_steps - 1)) * goal_reward

        total += g

    return total / n_simulations


def preferred_is_safe(
    nrows: int, ncols: int, wind_prob: float,
    goal_reward: float, cliff_reward: float, gamma: float,
    value_func: CPTValueFunction, use_cpt: bool,
) -> bool:
    """Check whether the globally preferred row is safe (row <= nrows-3)."""
    safe_threshold = nrows - 3
    values = {}
    for row in range(nrows - 1):
        if use_cpt:
            values[row] = discounted_cpt(row, nrows, ncols, wind_prob,
                                         goal_reward, cliff_reward, gamma, value_func)
        else:
            values[row] = discounted_ev(row, nrows, ncols, wind_prob,
                                        goal_reward, cliff_reward, gamma)
    preferred = max(values, key=values.get)
    return preferred <= safe_threshold


def find_threshold(
    nrows: int, ncols: int, wind_prob: float, cliff_reward: float,
    gamma: float, value_func: CPTValueFunction,
    use_cpt: bool = False,
    goal_lo: float = 1.0, goal_hi: float = 100000.0,
    tol: float = 0.1,
) -> float:
    """Binary search for goal_reward where the preferred row flips safe→risky.

    Returns the threshold goal_reward below which the globally preferred row
    becomes risky (row > nrows-3).
    """
    # Check that transition exists
    safe_at_hi = preferred_is_safe(nrows, ncols, wind_prob, goal_hi,
                                   cliff_reward, gamma, value_func, use_cpt)
    safe_at_lo = preferred_is_safe(nrows, ncols, wind_prob, goal_lo,
                                   cliff_reward, gamma, value_func, use_cpt)

    if not safe_at_hi:
        return None  # Safe never preferred even at high goal
    if safe_at_lo:
        return None  # Risky never preferred even at low goal

    # Binary search
    while (goal_hi - goal_lo) > tol:
        mid = (goal_lo + goal_hi) / 2
        if preferred_is_safe(nrows, ncols, wind_prob, mid,
                             cliff_reward, gamma, value_func, use_cpt):
            goal_hi = mid
        else:
            goal_lo = mid

    return (goal_lo + goal_hi) / 2


def analyze_config(
    shape: list, gamma: float, wind_prob: float, cliff_reward: float,
) -> dict | None:
    """Analyze a shared config to find EV and CPT thresholds and divergence window."""
    nrows, ncols = shape
    value_func = CPTValueFunction()

    ev_threshold = find_threshold(
        nrows, ncols, wind_prob, cliff_reward, gamma, value_func,
        use_cpt=False,
    )
    cpt_threshold = find_threshold(
        nrows, ncols, wind_prob, cliff_reward, gamma, value_func,
        use_cpt=True,
    )

    if ev_threshold is None or cpt_threshold is None:
        return None

    # Window: CPT threshold < goal < EV threshold → EV risky, CPT safe
    if cpt_threshold >= ev_threshold:
        return None  # No divergence window

    window = ev_threshold - cpt_threshold
    relative_window = window / ev_threshold

    if relative_window < 0.01:
        return None  # Window too narrow

    # Pick divergent goal at 50% into window (middle) for max margins
    divergent_goal = cpt_threshold + 0.5 * window
    # Pick baseline goal at 5x EV threshold so safe wins by wide margin
    baseline_goal = ev_threshold * 5.0

    safe_threshold = nrows - 3

    # Compute all row values at divergent goal
    ev_vals = {}
    cpt_vals = {}
    for row in range(nrows - 1):
        ev_vals[row] = discounted_ev(row, nrows, ncols, wind_prob,
                                     divergent_goal, cliff_reward, gamma)
        cpt_vals[row] = discounted_cpt(row, nrows, ncols, wind_prob,
                                       divergent_goal, cliff_reward, gamma, value_func)

    ev_preferred = max(ev_vals, key=ev_vals.get)
    cpt_preferred = max(cpt_vals, key=cpt_vals.get)

    # Verify: EV must prefer risky, CPT must prefer safe
    if ev_preferred <= safe_threshold or cpt_preferred > safe_threshold:
        return None

    # Best safe and best risky for each framework
    best_safe_ev = max((v for r, v in ev_vals.items() if r <= safe_threshold),
                       default=0)
    best_safe_cpt = max((v for r, v in cpt_vals.items() if r <= safe_threshold),
                        default=0)

    ev_margin = ev_vals[ev_preferred] - best_safe_ev  # positive = risky wins EV
    cpt_margin = best_safe_cpt - cpt_vals[ev_preferred]  # positive = safe wins CPT

    if ev_margin <= 0 or cpt_margin <= 0:
        return None

    # Estimate SNR: margin / approximate episode return std
    risky_row = ev_preferred
    p_cliff_risky = cliff_fall_probability(risky_row, nrows, ncols, wind_prob)
    steps_risky = path_steps(risky_row, nrows, ncols)
    g_success = (gamma ** (steps_risky - 1)) * divergent_goal
    steps_cliff = (nrows - 1 - risky_row) + (ncols - 1) // 2
    g_cliff = (gamma ** (max(steps_cliff, 1) - 1)) * cliff_reward
    episode_std = abs(g_success - g_cliff) * np.sqrt(
        p_cliff_risky * (1 - p_cliff_risky)
    )
    snr = ev_margin / max(episode_std, 1e-10)

    # Episodes needed to detect signal (approximate)
    episodes_needed = int(np.ceil(1.0 / (snr ** 2))) if snr > 0 else float('inf')

    return {
        'shape': shape,
        'gamma': gamma,
        'wind_prob': wind_prob,
        'cliff_reward': cliff_reward,
        'ev_threshold': ev_threshold,
        'cpt_threshold': cpt_threshold,
        'window': window,
        'relative_window': relative_window,
        'baseline_goal': round(baseline_goal),
        'divergent_goal': round(divergent_goal),
        'ev_margin': ev_margin,
        'cpt_margin': cpt_margin,
        'snr': snr,
        'episodes_needed': episodes_needed,
    }


def grid_search():
    """Search parameter space for configs with EV-risky, CPT-safe divergence."""
    shapes = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [4, 6]]
    gammas = np.arange(0.80, 0.96, 0.005)
    wind_probs = np.arange(0.05, 0.20, 0.01)
    cliff_rewards = [50, 100, 200, 500, 750, 1000]

    candidates = []
    total = 0

    for shape, gamma, wind_prob, cliff_reward in product(
        shapes, gammas, wind_probs, cliff_rewards
    ):
        total += 1
        result = analyze_config(shape, gamma, wind_prob, cliff_reward)
        if result is not None:
            candidates.append(result)

    print(f"Searched {total} configurations, found {len(candidates)} candidates.")
    return candidates


def print_result(result: dict, mc_validate: bool = False):
    """Pretty print a search result with per-row breakdown for both goals."""
    nrows, ncols = result['shape']
    value_func = CPTValueFunction()

    print(f"\n{'=' * 70}")
    print(f"Config: shape={result['shape']}, gamma={result['gamma']:.3f}, "
          f"wind={result['wind_prob']:.2f}, cliff_reward={result['cliff_reward']}")
    print(f"  EV threshold:  {result['ev_threshold']:.1f}")
    print(f"  CPT threshold: {result['cpt_threshold']:.1f}")
    print(f"  Window: [{result['cpt_threshold']:.1f}, {result['ev_threshold']:.1f}] "
          f"({result['relative_window']:.1%} of EV threshold)")
    print(f"  SNR: {result['snr']:.4f} "
          f"(~{result['episodes_needed']} episodes/batch to detect)")

    for label, goal in [("BASELINE", result['baseline_goal']),
                        ("DIVERGENT", result['divergent_goal'])]:
        print(f"\n  --- {label} (goal_reward={goal}) ---")
        print(f"  {'Row':<6} {'Steps':<7} {'P(cliff)':<10} "
              f"{'G_succ':<12} {'G_cliff':<12} {'v(G_s)':<12} {'v(G_c)':<12} "
              f"{'EV':<12} {'CPT':<12}")
        print(f"  {'-' * 95}")

        ev_values = {}
        cpt_values = {}
        for row in range(nrows - 1):
            steps = path_steps(row, nrows, ncols)
            p_cliff = cliff_fall_probability(row, nrows, ncols, result['wind_prob'])
            p_success = 1 - p_cliff

            g_success = (result['gamma'] ** (steps - 1)) * goal
            steps_cliff = (nrows - 1 - row) + (ncols - 1) // 2
            steps_cliff = max(steps_cliff, 1)
            g_cliff = (result['gamma'] ** (steps_cliff - 1)) * result['cliff_reward']

            ev = p_success * g_success + p_cliff * g_cliff
            cpt = p_success * value_func(g_success) + p_cliff * value_func(g_cliff)
            ev_values[row] = ev
            cpt_values[row] = cpt

            safe_marker = " <-- SAFE" if row <= nrows - 3 else ""
            print(f"  {row:<6} {steps:<7} {p_cliff:<10.4f} "
                  f"{g_success:<12.2f} {g_cliff:<12.2f} "
                  f"{value_func(g_success):<12.4f} {value_func(g_cliff):<12.4f} "
                  f"{ev:<12.4f} {cpt:<12.4f}{safe_marker}")

        ev_pref = max(ev_values, key=ev_values.get)
        cpt_pref = max(cpt_values, key=cpt_values.get)
        ev_label = "SAFE" if ev_pref <= nrows - 3 else "RISKY"
        cpt_label = "SAFE" if cpt_pref <= nrows - 3 else "RISKY"
        print(f"  EV prefers row {ev_pref} ({ev_label}), "
              f"CPT prefers row {cpt_pref} ({cpt_label})")

    if mc_validate:
        print(f"\n  Monte Carlo validation (50k sims, divergent goal={result['divergent_goal']}):")
        for row in [0, nrows - 2]:
            mc_ev = monte_carlo_validate(
                row, nrows, ncols, result['wind_prob'],
                result['divergent_goal'], result['cliff_reward'], result['gamma'],
            )
            analytical = discounted_ev(
                row, nrows, ncols, result['wind_prob'],
                result['divergent_goal'], result['cliff_reward'], result['gamma'],
            )
            print(f"    Row {row}: MC_EV={mc_ev:.4f} (analytical: {analytical:.4f})")


def generate_yaml(result: dict, goal_reward: int, extra: dict = None) -> str:
    """Generate YAML config string."""
    lines = [
        f"shape: {result['shape']}",
        f"stochasticity: windy",
        f"reward_cliff: {result['cliff_reward']}",
        f"reward_step: 0",
        f"reward_goal: {goal_reward}",
        f"wind_prob: {result['wind_prob']:.2f}",
        f"gamma: {result['gamma']:.3f}",
        f"baseline_type: ema",
        f"baseline_type_cpt: min",
        f"timesteps: 1000000",
        f"n_eval_episodes: 4",
        f"lr: 0.001",
        f"batch_size: 1024",
        f"entropy_coef: 1.0",
        f"entropy_coef_final: 0.01",
        f"n_seeds: 4",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    return '\n'.join(lines) + '\n'


def main():
    print("=" * 70)
    print("Grid Search: Finding Divergent Configs (EV-risky, CPT-safe)")
    print("Target: EV prefers risky path, CPT prefers safe path")
    print("Method: vary goal_reward to exploit CPT's diminishing sensitivity")
    print("=" * 70)

    candidates = grid_search()

    if not candidates:
        print("\nNo configurations found!")
        print("Try expanding the search space or relaxing constraints.")
        return None

    # Sort by SNR (most learnable first)
    candidates.sort(key=lambda r: -r['snr'])

    print(f"\n{'=' * 70}")
    print("Top 10 Candidates (sorted by SNR - most learnable first)")
    print("=" * 70)

    for result in candidates[:10]:
        print_result(result)

    # Pick best: highest SNR
    best = candidates[0]

    print(f"\n{'=' * 70}")
    print("BEST CONFIG (highest SNR)")
    print("=" * 70)
    print_result(best, mc_validate=True)

    # Generate and save configs
    config_dir = Path(__file__).parent.parent / "configs"

    # Baseline: both EV and CPT prefer safe
    baseline_yaml = generate_yaml(best, best['baseline_goal'])
    baseline_path = config_dir / "low_sensitivity_divergent_baseline.yaml"
    baseline_path.write_text(baseline_yaml)
    print(f"\nBaseline config written to {baseline_path}")
    print(f"  goal_reward={best['baseline_goal']} → both EV and CPT prefer safe")

    # Divergent: EV risky, CPT safe
    divergent_yaml = generate_yaml(best, best['divergent_goal'])
    divergent_path = config_dir / "low_sensitivity_divergent.yaml"
    divergent_path.write_text(divergent_yaml)
    print(f"\nDivergent config written to {divergent_path}")
    print(f"  goal_reward={best['divergent_goal']} → EV prefers risky, CPT prefers safe")

    print(f"\n{'=' * 70}")
    print("Recommended usage:")
    print(f"  python main.py --config low_sensitivity_divergent_baseline")
    print(f"  python main.py --config low_sensitivity_divergent")
    print("=" * 70)

    return best


if __name__ == "__main__":
    main()
