"""Utilities for CliffWalking RL experiments."""

from pathlib import Path
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from PIL import Image


def save_gif(frames, path, fps=4):
    """Save frames (numpy arrays) as animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0
    )
    return path


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


class CPTRewardWrapper(gym.RewardWrapper):
    """Gymnasium wrapper that transforms rewards using CPT value function."""

    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
    ):
        super().__init__(env)
        self.cpt_value = CPTValueFunction(alpha, beta, lambda_, reference_point)

    def reward(self, reward: SupportsFloat) -> float:
        return self.cpt_value(float(reward))


# =============================================================================
# LLM Agent prompt and tooling
# =============================================================================

CLIFFWALKING_PROMPT = """You are an RL agent navigating a 4x12 grid world (CliffWalking-v0).

LAYOUT:
- Grid: 4 rows (0-3) x 12 columns (0-11)
- START: Position 36 (row 3, col 0) - bottom-left
- GOAL: Position 47 (row 3, col 11) - bottom-right
- CLIFF: Positions 37-46 (row 3, cols 1-10) - bottom row between start/goal

ACTIONS:
- 0: UP (row decreases)
- 1: RIGHT (column increases)
- 2: DOWN (row increases)
- 3: LEFT (column decreases)

REWARDS:
- Each step: -1
- Falling off cliff: -100, return to start
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


def format_cliffwalking_state(state: int) -> str:
    """Format CliffWalking state to human-readable description."""
    row = state // 12
    col = state % 12

    desc = f"Position: Row {row}, Column {col}"

    if state == 36:
        desc += " (START)"
    elif state == 47:
        desc += " (GOAL)"
    elif 37 <= state <= 46:
        desc += " (ON CLIFF!)"
    elif row == 3:
        desc += " (bottom row, adjacent to cliff)"
    elif row == 0:
        desc += " (top row, safe)"

    goal_row, goal_col = 3, 11
    desc += f"\nDistance to goal: {abs(row - goal_row) + abs(col - goal_col)} steps"

    return desc
