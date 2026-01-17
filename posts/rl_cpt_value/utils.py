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


FROZENLAKE_PROMPT = """You are navigating a 4x4 frozen lake grid (FrozenLake-v1).

LAYOUT (S=Start, F=Frozen, H=Hole, G=Goal):
     Col0 Col1 Col2 Col3
Row0:  S    F    F    F
Row1:  F    H    F    H
Row2:  F    F    F    H
Row3:  H    F    F    G

HOLES are at: (1,1), (1,3), (2,3), (3,0)

ACTIONS (Gymnasium FrozenLake mapping):
- 0: LEFT (column decreases)
- 1: DOWN (row increases)
- 2: RIGHT (column increases)
- 3: UP (row decreases)

CRITICAL - SLIPPERY ICE MECHANICS:
The ice is slippery! When you choose an action, there's only a 1/3 chance you go that direction.
You have a 1/3 chance of slipping to each perpendicular direction instead.
Example: If you choose DOWN, you might go DOWN (1/3), LEFT (1/3), or RIGHT (1/3).

SAFE NAVIGATION STRATEGY:
- Walls are SAFE: If you would slide off the grid, you just stay in place.
- Before moving, consider: "If I slip perpendicular, will I fall in a hole?"
- Positions next to holes on multiple sides are DANGEROUS.

DANGEROUS POSITIONS (adjacent to holes):
- (0,1): Hole below at (1,1)
- (1,0): Hole right at (1,1)
- (1,2): Holes left (1,1) and right (1,3)
- (0,3): Hole below at (1,3)
- (2,2): Hole right at (2,3)
- (1,3): IS A HOLE
- (2,3): IS A HOLE
- (3,1): Hole left at (3,0)

SAFER PATH CONCEPT:
From Start (0,0), going RIGHT along top row keeps wall above (safe from upward slips).
From (0,2), going DOWN through (1,2), (2,2) to reach (2,1) or (3,1) then RIGHT to Goal.
Avoid positions where perpendicular slips lead to holes.

REWARDS:
- Reaching goal: +1
- All other steps: 0
- Falling in hole: episode ends (0 reward)

OBJECTIVE: Reach the Goal (G) at position (3,3) without falling in any Hole (H).

Think carefully about slip risks before choosing your action."""


FROZENLAKE_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "select_action",
        "description": "Select the next action to take in FrozenLake",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning about current position, slip risks, and why this action is best",
                },
                "action": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3],
                    "description": "Action to take: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP",
                },
            },
            "required": ["reasoning", "action"],
            "additionalProperties": False,
        },
    },
}

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


# =============================================================================
# Helper functions
# =============================================================================


def compute_action_entropy(actions):
    """Compute normalized entropy of action distribution (0-1 scale)."""
    if not actions:
        return 0.0
    counts = np.bincount(actions, minlength=4)
    probs = counts / len(actions)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(4)
    return round(entropy / max_entropy, 3)


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


def format_frozenlake_state(state: int) -> str:
    """Format FrozenLake state with visual grid and safety analysis."""
    row = state // 4
    col = state % 4

    grid = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]
    tile = grid[row][col]
    tile_names = {"S": "START", "F": "FROZEN", "H": "HOLE", "G": "GOAL"}

    visual = "Current grid (you are [X]):\n"
    visual += "     Col0 Col1 Col2 Col3\n"
    for r in range(4):
        visual += f"Row{r}: "
        for c in range(4):
            if r == row and c == col:
                visual += "[X]  "
            else:
                visual += f" {grid[r][c]}   "
        visual += "\n"

    desc = f"Position: Row {row}, Column {col} ({tile_names[tile]})\n\n{visual}"

    holes = {(1, 1), (1, 3), (2, 3), (3, 0)}

    desc += "\nACTION SAFETY ANALYSIS:\n"
    actions = {
        0: ("LEFT", (row, col - 1)),
        1: ("DOWN", (row + 1, col)),
        2: ("RIGHT", (row, col + 1)),
        3: ("UP", (row - 1, col)),
    }

    for action_id, (action_name, (new_r, new_c)) in actions.items():
        if new_r < 0 or new_r > 3 or new_c < 0 or new_c > 3:
            desc += f"  {action_id}={action_name}: WALL (you stay in place - SAFE)\n"
        elif (new_r, new_c) in holes:
            desc += f"  {action_id}={action_name}: HOLE at ({new_r},{new_c}) - DANGER!\n"
        elif (new_r, new_c) == (3, 3):
            desc += f"  {action_id}={action_name}: GOAL at ({new_r},{new_c}) - WIN!\n"
        else:
            desc += f"  {action_id}={action_name}: Frozen at ({new_r},{new_c})\n"

    desc += "\nSLIP DANGER (perpendicular moves if you slip):\n"
    perpendicular = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2],
    }
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

    for action_id in range(4):
        slip_dangers = []
        for slip_action in perpendicular[action_id]:
            _, (slip_r, slip_c) = actions[slip_action]
            if 0 <= slip_r <= 3 and 0 <= slip_c <= 3 and (slip_r, slip_c) in holes:
                slip_dangers.append(f"{action_names[slip_action]} into hole at ({slip_r},{slip_c})")
        if slip_dangers:
            desc += f"  If you choose {action_id}={action_names[action_id]}, you might slip: {', '.join(slip_dangers)}\n"

    desc += f"\nDistance to goal: {abs(row - 3) + abs(col - 3)} steps (Manhattan)"

    return desc
