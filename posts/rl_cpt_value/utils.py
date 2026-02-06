"""Utilities for CliffWalking RL experiments."""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


def load_config(config_name: str = "base"):
    """Load configuration from YAML file, inheriting defaults from base.yaml."""
    config_dir = Path(__file__).parent / "configs"
    with open(config_dir / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    if config_name != "base":
        with open(config_dir / f"{config_name}.yaml") as f:
            cfg.update(yaml.safe_load(f))
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

    def w_prime_plus(self, p: float, eps: float = 1e-7) -> float:
        """Numerical derivative of w_plus at p."""
        if p <= eps:
            return self.w_plus(eps) / eps
        if p >= 1 - eps:
            return (1.0 - self.w_plus(1 - eps)) / eps
        return (self.w_plus(p + eps) - self.w_plus(p - eps)) / (2 * eps)

    def w_prime_minus(self, p: float, eps: float = 1e-7) -> float:
        """Numerical derivative of w_minus at p."""
        if p <= eps:
            return self.w_minus(eps) / eps
        if p >= 1 - eps:
            return (1.0 - self.w_minus(1 - eps)) / eps
        return (self.w_minus(p + eps) - self.w_minus(p - eps)) / (2 * eps)


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
