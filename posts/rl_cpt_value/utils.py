"""Utilities for CliffWalking RL experiments."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from PIL import Image


def load_config(config_name: str = "base"):
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "configs" / f"{config_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


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
