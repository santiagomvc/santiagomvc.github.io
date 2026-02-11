"""Utilities for CliffWalking RL experiments."""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_name: str = "base"):
    """Load configuration from YAML file, inheriting defaults from base.yaml."""
    config_dir = Path(__file__).parent / "configs"
    with open(config_dir / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    if config_name != "base":
        with open(config_dir / f"{config_name}.yaml") as f:
            override = yaml.safe_load(f)
            if override:
                _deep_merge(cfg, override)
    return cfg


def save_episodes_gif(episodes_frames, output_path, cols=2, fps=4):
    """Save multiple episodes as a combined grid GIF.

    Args:
        episodes_frames: List of episodes, each episode is a list of frames (numpy arrays).
        output_path: Path for the output GIF.
        cols: Number of columns in the grid.
        fps: Frames per second.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy frames to PIL images
    all_frames = [[Image.fromarray(f) for f in ep] for ep in episodes_frames]

    # Pad shorter episodes by repeating last frame
    max_frames = max(len(ep) for ep in all_frames)
    for ep in all_frames:
        while len(ep) < max_frames:
            ep.append(ep[-1].copy())

    # Calculate grid dimensions
    n_eps = len(all_frames)
    rows = (n_eps + cols - 1) // cols
    w, h = all_frames[0][0].size

    # Create combined frames
    combined = []
    for i in range(max_frames):
        grid = Image.new("RGB", (cols * w, rows * h))
        for j, ep in enumerate(all_frames):
            grid.paste(ep[i].convert("RGB"), ((j % cols) * w, (j // cols) * h))
        combined.append(grid)

    combined[0].save(
        output_path, save_all=True, append_images=combined[1:],
        duration=int(1000 / fps), loop=0
    )
    return output_path


# =============================================================================
# CPT (Cumulative Prospect Theory) - Tversky & Kahneman (1992)
# =============================================================================


class CPTValueFunction:
    """CPT value function: S-shaped with loss aversion.

    v(x) = x^α               for x ≥ 0  (gains)
    v(x) = -λ * (-x)^β       for x < 0  (losses)

    Default params from Tversky & Kahneman (1992):
    - α = 0.88: diminishing sensitivity for gains
    - β = 0.88: diminishing sensitivity for losses
    - λ = 2.25: loss aversion coefficient
    """

    def __init__(
        self,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.reference_point = reference_point

    def __call__(self, outcome: float) -> float:
        x = outcome - self.reference_point
        if x >= 0:
            return np.power(x, self.alpha) if x > 0 else 0.0
        else:
            return -self.lambda_ * np.power(-x, self.beta)


class CPTWeightingFunction:
    """CPT probability weighting functions w+(p) and w-(p).

    Tversky & Kahneman 1992, equation 6.
    Inverse S-shape: overweights small p, underweights large p.

    w+(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)     γ = 0.61 (median)
    w-(p) = p^δ / (p^δ + (1-p)^δ)^(1/δ)     δ = 0.69 (median)
    """

    def __init__(self, gamma_plus: float = 0.61, gamma_minus: float = 0.69):
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus

    def w_plus(self, p: float) -> float:
        """Weighting function for gains."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        g = self.gamma_plus
        return p**g / (p**g + (1 - p)**g) ** (1 / g)

    def w_minus(self, p: float) -> float:
        """Weighting function for losses."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        d = self.gamma_minus
        return p**d / (p**d + (1 - p)**d) ** (1 / d)

    def w_prime_plus(self, p: float, eps: float = 1e-7, max_wprime: float = 50.0) -> float:
        """Numerical derivative of w_plus at p, clamped for Lipschitz bound."""
        if p <= eps:
            raw = self.w_plus(eps) / eps
        elif p >= 1 - eps:
            raw = (1.0 - self.w_plus(1 - eps)) / eps
        else:
            raw = (self.w_plus(p + eps) - self.w_plus(p - eps)) / (2 * eps)
        return min(raw, max_wprime)

    def w_prime_minus(self, p: float, eps: float = 1e-7, max_wprime: float = 50.0) -> float:
        """Numerical derivative of w_minus at p, clamped for Lipschitz bound."""
        if p <= eps:
            raw = self.w_minus(eps) / eps
        elif p >= 1 - eps:
            raw = (1.0 - self.w_minus(1 - eps)) / eps
        else:
            raw = (self.w_minus(p + eps) - self.w_minus(p - eps)) / (2 * eps)
        return min(raw, max_wprime)


class PerPathSlidingWindowCPT:
    """CPT decision weights computed per-path instead of per-batch.

    Uses pre-computed per-row cliff probabilities from env config to determine
    the correct decision weight for each episode based on its path's success rate.

    This fixes the batch-level weighting issue where risky success gets
    overweighted (rare extreme in batch) instead of underweighted (likely
    outcome on its path).
    """

    def __init__(
        self,
        weighting_func: CPTWeightingFunction,
        env_config: dict,
        gamma: float,
        max_batches: int = 5,
        decay: float = 0.8,
        reference_point: float = 0.0,
    ):
        self.weighting_func = weighting_func
        self.max_batches = max_batches
        self.decay = decay
        self.reference_point = reference_point
        self.buffer = []

        from path_likelihood import cliff_fall_probability

        # Pre-compute per-row success rates and canonical returns
        env = env_config['env']
        nrows, ncols = env['shape']
        wind_prob = env.get('wind_prob', 0.0)
        goal_reward = env['reward_goal']

        self.path_info = {}  # {row: (canonical_return, p_success)}
        for row in range(nrows - 1):
            steps = 2 * (nrows - 1 - row) + (ncols - 1)
            p_cliff = cliff_fall_probability(row, nrows, ncols, wind_prob)
            canonical_return = (gamma ** (steps - 1)) * goal_reward
            self.path_info[row] = (canonical_return, 1.0 - p_cliff)

        # Cliff threshold: returns below this are cliff outcomes
        min_success_return = min(r for r, _ in self.path_info.values())
        self.cliff_threshold = min_success_return * 0.5

    def add_batch(self, episode_returns: list[float]):
        """Add a batch of episode returns (kept for interface compatibility)."""
        self.buffer = [(r, w * self.decay) for r, w in self.buffer]
        self.buffer.append((np.array(episode_returns), 1.0))
        if len(self.buffer) > self.max_batches:
            self.buffer = self.buffer[-self.max_batches:]

    def _identify_row(self, episode_return: float) -> int | None:
        """Identify which row an episode traversed from its return value."""
        if episode_return < self.cliff_threshold:
            return None  # Cliff episode

        best_row = None
        best_ratio = float('inf')
        for row, (canonical, _) in self.path_info.items():
            if canonical > 0:
                ratio = abs(np.log(episode_return / canonical))
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_row = row
        return best_row

    def compute_decision_weights(self, episode_returns: list[float]) -> np.ndarray:
        """Compute CPT decision weights using per-path success rates."""
        n = len(episode_returns)
        returns = np.array(episode_returns)
        weights = np.zeros(n)

        for i in range(n):
            row = self._identify_row(returns[i])

            if row is not None:
                _, p_success = self.path_info[row]
                weights[i] = self.weighting_func.w_plus(p_success)
            else:
                riskiest_row = max(self.path_info.keys())
                _, p_success = self.path_info[riskiest_row]
                weights[i] = 1.0 - self.weighting_func.w_plus(p_success)

        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights * (n / weight_sum)

        return weights


class SlidingWindowCPT:
    """Sliding window for estimating return distribution and computing CPT decision weights.

    Maintains a buffer of recent batch returns with exponential decay to estimate
    the empirical CDF, then computes cumulative CPT decision weights for each
    episode in the current batch.
    """

    def __init__(
        self,
        weighting_func: CPTWeightingFunction,
        max_batches: int = 5,
        decay: float = 0.8,
        reference_point: float = 0.0,
    ):
        self.weighting_func = weighting_func
        self.max_batches = max_batches
        self.decay = decay
        self.reference_point = reference_point
        self.buffer = []  # list of (returns_array, weight) tuples

    def add_batch(self, episode_returns: list[float]):
        """Add a batch of episode returns and decay old batches."""
        # Decay existing weights
        self.buffer = [(r, w * self.decay) for r, w in self.buffer]
        # Add new batch with weight 1.0
        self.buffer.append((np.array(episode_returns), 1.0))
        # Trim to max_batches
        if len(self.buffer) > self.max_batches:
            self.buffer = self.buffer[-self.max_batches:]

    def _get_weighted_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all samples and their weights from the sliding window."""
        all_returns = []
        all_weights = []
        for returns, batch_weight in self.buffer:
            all_returns.extend(returns)
            all_weights.extend([batch_weight] * len(returns))
        return np.array(all_returns), np.array(all_weights)

    def compute_decision_weights(self, episode_returns: list[float]) -> np.ndarray:
        """Compute CPT decision weights for current batch episodes.

        Uses the sliding window to estimate the empirical CDF, then applies
        cumulative probability weighting separately for gains and losses.

        For gains (sorted increasing, decumulative):
            π_i = w+(P(R >= r_i)) - w+(P(R > r_i))
        For losses (sorted increasing, cumulative):
            π_i = w-(P(R <= r_i)) - w-(P(R < r_i))

        Returns weights in the original episode order.
        """
        returns = np.array(episode_returns)
        n = len(returns)

        if n == 0:
            return np.array([])

        # Get all historical samples for CDF estimation
        all_returns, all_weights = self._get_weighted_samples()
        total_weight = all_weights.sum()

        if total_weight == 0:
            return np.ones(n) / n

        # Sort current batch returns and track original indices
        sorted_indices = np.argsort(returns)
        sorted_returns = returns[sorted_indices]

        # Compute decision weights, handling ties by sharing weight equally
        decision_weights = np.zeros(n)

        # Group by unique values to handle ties
        unique_returns = np.unique(sorted_returns)

        for r_i in unique_returns:
            x_i = r_i - self.reference_point
            tie_mask = sorted_returns == r_i
            n_ties = tie_mask.sum()

            if x_i >= 0:
                # Gains: decumulative weighting
                p_geq = np.sum(all_weights[all_returns >= r_i]) / total_weight
                p_gt = np.sum(all_weights[all_returns > r_i]) / total_weight
                w_geq = self.weighting_func.w_plus(p_geq)
                w_gt = self.weighting_func.w_plus(p_gt)
                rank_weight = (w_geq - w_gt) / n_ties
            else:
                # Losses: cumulative weighting
                p_leq = np.sum(all_weights[all_returns <= r_i]) / total_weight
                p_lt = np.sum(all_weights[all_returns < r_i]) / total_weight
                w_leq = self.weighting_func.w_minus(p_leq)
                w_lt = self.weighting_func.w_minus(p_lt)
                rank_weight = (w_leq - w_lt) / n_ties

            decision_weights[tie_mask] = rank_weight

        # Normalize to sum to n (so average weight is 1.0, preserving gradient scale)
        weight_sum = decision_weights.sum()
        if weight_sum > 0:
            decision_weights = decision_weights * (n / weight_sum)

        # Map back to original episode order
        result = np.zeros(n)
        result[sorted_indices] = decision_weights
        return result


class PerStepSlidingWindowCPT:
    """Per-timestep CPT decision weights from the distribution of G_t at each t.

    For each timestep position t, gathers all G_t values across episodes that
    are still active at t, then applies CPT probability weighting to that
    per-timestep distribution. This is the correct per-step CPT formulation:
    it treats {G_t^(1), G_t^(2), ..., G_t^(N)} as a prospect at each t.
    """

    def __init__(
        self,
        weighting_func: CPTWeightingFunction,
        max_batches: int = 5,
        decay: float = 0.8,
        reference_point: float = 0.0,
    ):
        self.weighting_func = weighting_func
        self.max_batches = max_batches
        self.decay = decay
        self.reference_point = reference_point
        self.buffer = []  # list of (per_step_returns_list, weight, metadata)

    def add_batch(self, batch_per_step_returns: list[list[float]], episode_metadata: list):
        """Add a batch where each element is [G_0, G_1, ..., G_T] for one episode.

        Args:
            batch_per_step_returns: Per-step returns for each episode.
            episode_metadata: Opaque metadata per episode, passed to is_ratio_fn later.
        """
        self.buffer = [(data, w * self.decay, meta) for data, w, meta in self.buffer]
        self.buffer.append((batch_per_step_returns, 1.0, episode_metadata))
        if len(self.buffer) > self.max_batches:
            self.buffer = self.buffer[-self.max_batches:]

    def compute_decision_weights(self, batch_per_step_returns: list[list[float]], is_ratio_fn) -> list[list[float]]:
        """Compute π_{i,t} for each episode i, timestep t.

        For each timestep t, gather all G_t values from episodes that are
        still active at t (episode length > t). Apply SlidingWindowCPT-style
        weighting to that per-timestep distribution.

        Returns list of lists: weights[i][t] = π_{i,t}
        """
        n_episodes = len(batch_per_step_returns)
        if n_episodes == 0:
            return []

        max_T = max(len(ep) for ep in batch_per_step_returns)

        # Initialize output weights
        weights = [[] for _ in range(n_episodes)]

        for t in range(max_T):
            # Gather all G_t values from buffer + current batch for episodes active at t
            all_returns_t = []
            all_sample_weights = []
            for buf_idx, (buf_data, buf_weight, buf_meta) in enumerate(self.buffer):
                is_current_batch = (buf_idx == len(self.buffer) - 1)
                for ep_idx, ep in enumerate(buf_data):
                    if t < len(ep):
                        sample_weight = buf_weight
                        if not is_current_batch:
                            sample_weight *= is_ratio_fn(buf_meta[ep_idx])
                        all_returns_t.append(ep[t])
                        all_sample_weights.append(sample_weight)

            all_returns_t = np.array(all_returns_t) if all_returns_t else np.array([])
            all_sample_weights = np.array(all_sample_weights) if all_sample_weights else np.array([])
            total_weight = all_sample_weights.sum() if len(all_sample_weights) > 0 else 0.0

            # Collect current batch indices active at this timestep
            active_indices = [i for i in range(n_episodes) if t < len(batch_per_step_returns[i])]
            n_active = len(active_indices)

            if n_active == 0:
                continue

            if total_weight == 0:
                # No historical data, uniform weights
                for i in active_indices:
                    weights[i].append(1.0)
                continue

            # Get current batch returns at timestep t for active episodes
            current_returns_t = np.array([batch_per_step_returns[i][t] for i in active_indices])

            # Compute per-timestep decision weights using CPT weighting
            # Same gain/loss separation logic as SlidingWindowCPT
            sorted_indices = np.argsort(current_returns_t)
            sorted_returns = current_returns_t[sorted_indices]

            decision_weights_t = np.zeros(n_active)
            unique_returns = np.unique(sorted_returns)

            for r_i in unique_returns:
                x_i = r_i - self.reference_point
                tie_mask = sorted_returns == r_i
                n_ties = tie_mask.sum()

                if x_i >= 0:
                    # Gains: decumulative weighting
                    p_geq = np.sum(all_sample_weights[all_returns_t >= r_i]) / total_weight
                    p_gt = np.sum(all_sample_weights[all_returns_t > r_i]) / total_weight
                    w_geq = self.weighting_func.w_plus(p_geq)
                    w_gt = self.weighting_func.w_plus(p_gt)
                    rank_weight = (w_geq - w_gt) / n_ties
                else:
                    # Losses: cumulative weighting
                    p_leq = np.sum(all_sample_weights[all_returns_t <= r_i]) / total_weight
                    p_lt = np.sum(all_sample_weights[all_returns_t < r_i]) / total_weight
                    w_leq = self.weighting_func.w_minus(p_leq)
                    w_lt = self.weighting_func.w_minus(p_lt)
                    rank_weight = (w_leq - w_lt) / n_ties

                decision_weights_t[tie_mask] = rank_weight

            # Normalize so mean weight = 1.0 among active episodes at this t
            weight_sum = decision_weights_t.sum()
            if weight_sum > 0:
                decision_weights_t = decision_weights_t * (n_active / weight_sum)

            # Map back to original episode order
            result_t = np.zeros(n_active)
            result_t[sorted_indices] = decision_weights_t

            for idx_pos, i in enumerate(active_indices):
                weights[i].append(float(result_t[idx_pos]))

        return weights


# =============================================================================
# LLM Agent prompt and tooling
# =============================================================================

def get_cliffwalking_prompt(shape: tuple[int, int], reward_cliff: float, reward_step: float) -> str:
    """Generate CliffWalking prompt for configurable grid size and rewards."""
    nrows, ncols = shape
    start_pos = (nrows - 1) * ncols
    goal_pos = nrows * ncols - 1
    cliff_start = start_pos + 1
    cliff_end = goal_pos - 1

    return f"""You are an RL agent navigating a {nrows}x{ncols} grid world (CliffWalking).

LAYOUT:
- Grid: {nrows} rows (0-{nrows-1}) x {ncols} columns (0-{ncols-1})
- START: Position {start_pos} (row {nrows-1}, col 0) - bottom-left
- GOAL: Position {goal_pos} (row {nrows-1}, col {ncols-1}) - bottom-right
- CLIFF: Positions {cliff_start}-{cliff_end} (row {nrows-1}, cols 1-{ncols-2}) - bottom row between start/goal

ACTIONS:
- 0: UP (row decreases)
- 1: RIGHT (column increases)
- 2: DOWN (row increases)
- 3: LEFT (column decreases)

REWARDS:
- Each step: {reward_step}
- Falling off cliff: {reward_cliff}, return to start
- Reaching goal: episode ends

OBJECTIVE: Reach the goal with minimum total penalty. The safe path goes UP from start, RIGHT across top rows, then DOWN to goal.

Think step-by-step before choosing your action."""


CLIFFWALKING_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "select_action",
        "description": "Select the next action to take in CliffWalking",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning about current position and why this action is best",
                },
                "action": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3],
                    "description": "Action to take: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT",
                },
            },
            "required": ["reasoning", "action"],
            "additionalProperties": False,
        },
    },
}


def format_cliffwalking_state(state: int, shape: tuple[int, int] = (4, 12)) -> str:
    """Format CliffWalking state to human-readable description."""
    nrows, ncols = shape
    row = state // ncols
    col = state % ncols

    desc = f"Position: Row {row}, Column {col}"

    start_state = (nrows - 1) * ncols  # Bottom-left
    goal_state = nrows * ncols - 1  # Bottom-right
    cliff_start = start_state + 1
    cliff_end = goal_state - 1

    if state == start_state:
        desc += " (START)"
    elif state == goal_state:
        desc += " (GOAL)"
    elif cliff_start <= state <= cliff_end and row == nrows - 1:
        desc += " (ON CLIFF!)"
    elif row == nrows - 1:
        desc += " (bottom row, adjacent to cliff)"
    elif row == 0:
        desc += " (top row, safe)"

    goal_row, goal_col = nrows - 1, ncols - 1
    desc += f"\nDistance to goal: {abs(row - goal_row) + abs(col - goal_col)} steps"

    return desc


def save_training_curves(history: dict, output_dir: str, agent_name: str, window: int = 100):
    """Save reward and loss curves as PNG files.

    Args:
        history: Dict with 'episode_rewards' and 'batch_losses' lists
        output_dir: Directory to save plots
        agent_name: Agent name for plot titles
        window: Window size for smoothing (default: 100)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reward plot
    rewards = history['episode_rewards']
    if len(rewards) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, color='blue', label=f'Smoothed (window={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f'{agent_name} - Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / 'rewards.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_path / 'rewards.png'}")

    # Loss plot
    losses = history['batch_losses']
    if len(losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(losses, alpha=0.3, color='red', label='Raw')
        loss_window = min(window, len(losses))
        if len(losses) >= loss_window:
            smoothed = np.convolve(losses, np.ones(loss_window)/loss_window, mode='valid')
            ax.plot(range(loss_window-1, len(losses)), smoothed, color='red', label=f'Smoothed (window={loss_window})')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{agent_name} - Batch Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / 'losses.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {output_path / 'losses.png'}")


def evaluate_paths(env, agent, n_episodes, config_name="base"):
    """Evaluate which path (row) a trained agent takes.

    Runs n_episodes, tracks the minimum row reached per episode
    (determines path), and counts cliff falls.

    Returns:
        dict with keys: "path_counts" ({row: count}), "cliff_falls" (int), "n_episodes" (int)
    """
    cfg = load_config(config_name)
    nrows, ncols = cfg["env"]["shape"]
    reward_cliff = cfg["env"]["reward_cliff"]

    path_counts = {}
    cliff_falls = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        min_row = nrows  # will be updated on first step
        fell_off_cliff = False

        for _ in range(500):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)

            row = state // ncols
            if row < min_row:
                min_row = row

            if reward == reward_cliff:
                fell_off_cliff = True

            if terminated or truncated:
                break

        if fell_off_cliff:
            cliff_falls += 1
        path_counts[min_row] = path_counts.get(min_row, 0) + 1

    return {"path_counts": path_counts, "cliff_falls": cliff_falls, "n_episodes": n_episodes}


def summarize_paths(all_path_results, nrows, agent_name, config_name):
    """Pool path results across seeds and print a compact single-line summary."""
    total_eps = sum(r["n_episodes"] for r in all_path_results)
    total_cliff = sum(r["cliff_falls"] for r in all_path_results)
    total_paths = {}
    for r in all_path_results:
        for row, count in r["path_counts"].items():
            total_paths[row] = total_paths.get(row, 0) + count

    path_parts = []
    for i in range(1, nrows):
        row = nrows - 1 - i
        pct = total_paths.get(row, 0) / total_eps * 100
        path_parts.append(f"Path{i}={pct:.0f}%")
    success_pct = (total_eps - total_cliff) / total_eps * 100
    cliff_pct = total_cliff / total_eps * 100
    print(f"{agent_name} ({config_name}): {' '.join(path_parts)} | Success={success_pct:.0f}% Cliff={cliff_pct:.0f}% [{total_eps} eps]")
