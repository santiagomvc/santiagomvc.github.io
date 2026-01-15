"""Compare evaluation results across agents."""

import json
from pathlib import Path


def load_stats(outputs_dir="outputs"):
    """Load all stats JSON files from outputs directory."""
    stats = {}
    for path in Path(outputs_dir).glob("*_stats.json"):
        with open(path) as f:
            data = json.load(f)
            stats[data["agent"]] = data
    return stats


def print_comparison(stats):
    """Print comparison table."""
    if not stats:
        print("No stats found in outputs/")
        return

    print("\n" + "=" * 70)
    print(f"{'Agent':<12} | {'Goal Rate':>10} | {'Avg Reward':>11} | {'Avg Cliffs':>11} | {'Safe Row %':>10}")
    print("-" * 70)

    for agent, data in sorted(stats.items()):
        eps = data["episodes"]
        n = len(eps)
        goals = sum(1 for e in eps if e["reached_goal"])
        avg_reward = sum(e["reward"] for e in eps) / n
        avg_cliffs = sum(e["cliff_falls"] for e in eps) / n
        avg_safe = sum(e["safe_row_pct"] for e in eps) / n

        print(f"{agent:<12} | {goals:>5}/{n:<4} | {avg_reward:>11.1f} | {avg_cliffs:>11.2f} | {avg_safe:>9.1f}%")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    stats = load_stats()
    print_comparison(stats)
