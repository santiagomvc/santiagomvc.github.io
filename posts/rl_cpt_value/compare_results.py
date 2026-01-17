"""Compare evaluation results across agents and environments."""

import json
from pathlib import Path


def load_stats(outputs_dir="outputs"):
    """Load all stats JSON files from outputs directory, grouped by environment."""
    by_env = {}
    for path in Path(outputs_dir).glob("*_stats.json"):
        with open(path) as f:
            data = json.load(f)
            env = data.get("env", "unknown")
            agent = data["agent"]
            by_env.setdefault(env, {})[agent] = data
    return by_env


def print_cliffwalking_table(stats):
    """Print CliffWalking comparison table."""
    print(f"{'Agent':<12} | {'Goal Rate':>10} | {'Avg Reward':>11} | {'Cliff Falls':>11} | {'Safe Row %':>10}")
    print("-" * 70)

    for agent, data in sorted(stats.items()):
        eps = data["episodes"]
        n = len(eps)
        goals = sum(1 for e in eps if e["reached_goal"])
        avg_reward = sum(e["reward"] for e in eps) / n
        avg_cliffs = sum(e["cliff_falls"] for e in eps) / n
        avg_safe = sum(e["safe_row_pct"] for e in eps) / n

        print(f"{agent:<12} | {goals:>5}/{n:<4} | {avg_reward:>11.1f} | {avg_cliffs:>11.2f} | {avg_safe:>9.1f}%")


def print_frozenlake_table(stats):
    """Print FrozenLake comparison table."""
    print(f"{'Agent':<12} | {'Goal Rate':>10} | {'Avg Reward':>11} | {'Holes':>11} | {'Safe Zone %':>11}")
    print("-" * 70)

    for agent, data in sorted(stats.items()):
        eps = data["episodes"]
        n = len(eps)
        goals = sum(1 for e in eps if e["reached_goal"])
        avg_reward = sum(e["reward"] for e in eps) / n
        holes = sum(1 for e in eps if e["fell_in_hole"])
        # Handle old stats without safe_zone_pct
        avg_safe = sum(e.get("safe_zone_pct", 0) for e in eps) / n

        print(f"{agent:<12} | {goals:>5}/{n:<4} | {avg_reward:>11.2f} | {holes:>6}/{n:<4} | {avg_safe:>10.1f}%")


def print_behavioral_table(stats, env_type):
    """Print behavioral metrics table for an environment."""
    print(f"\n{'Agent':<12} | {'Entropy':>8} | {'Path Eff':>8} | {'Action Distribution (%)':>30}")
    print("-" * 70)

    # Action labels differ by environment
    if "CliffWalking" in env_type:
        action_labels = "U/R/D/L"
    else:
        action_labels = "L/D/R/U"

    for agent, data in sorted(stats.items()):
        eps = data["episodes"]
        n = len(eps)

        # Average entropy
        avg_entropy = sum(e.get("action_entropy", 0) for e in eps) / n

        # Average path efficiency/directness
        if "CliffWalking" in env_type:
            avg_path = sum(e.get("path_directness", 0) for e in eps) / n
        else:
            avg_path = sum(e.get("path_efficiency", 0) for e in eps) / n

        # Average action distribution
        avg_dist = [0, 0, 0, 0]
        for e in eps:
            dist = e.get("action_dist", [0, 0, 0, 0])
            for i in range(4):
                avg_dist[i] += dist[i]
        avg_dist = [round(d / n, 1) for d in avg_dist]
        dist_str = f"{action_labels}: {avg_dist[0]:>4}/{avg_dist[1]:>4}/{avg_dist[2]:>4}/{avg_dist[3]:>4}"

        print(f"{agent:<12} | {avg_entropy:>8.3f} | {avg_path:>8.3f} | {dist_str:>30}")


def print_comparison(by_env):
    """Print comparison tables for all environments."""
    if not by_env:
        print("No stats found in outputs/")
        return

    for env, stats in sorted(by_env.items()):
        print("\n" + "=" * 70)
        print(f" {env}")
        print("=" * 70)

        if "CliffWalking" in env:
            print_cliffwalking_table(stats)
            print_behavioral_table(stats, env)
        elif "FrozenLake" in env:
            print_frozenlake_table(stats)
            print_behavioral_table(stats, env)
        else:
            print(f"Unknown environment: {env}")

    print()


if __name__ == "__main__":
    by_env = load_stats()
    print_comparison(by_env)
