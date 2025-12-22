"""Visualization utilities for CliffWalking."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Grid constants
NROWS, NCOLS = 4, 12
START, GOAL = 36, 47
CLIFF = set(range(37, 47))


def state_to_pos(state):
    """Convert state index to (row, col)."""
    return state // NCOLS, state % NCOLS


def draw_frame(ax, agent_pos, path_history, step, reward):
    """Draw one frame of the grid."""
    ax.clear()
    ax.set_xlim(-0.5, NCOLS - 0.5)
    ax.set_ylim(-0.5, NROWS - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Draw special cells
    for state, color, label in [(GOAL, "#2ecc71", "G"), (START, "#3498db", "S")]:
        row, col = state_to_pos(state)
        ax.add_patch(patches.Rectangle((col-0.5, row-0.5), 1, 1, facecolor=color, alpha=0.7))
        ax.text(col, row, label, ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw cliff
    for state in CLIFF:
        row, col = state_to_pos(state)
        ax.add_patch(patches.Rectangle((col-0.5, row-0.5), 1, 1, facecolor="#e74c3c", alpha=0.7))

    # Draw path
    if len(path_history) > 1:
        rows, cols = zip(*path_history)
        ax.plot(cols, rows, "o-", color="#9b59b6", alpha=0.5, markersize=4)

    # Draw agent
    ax.plot(agent_pos[1], agent_pos[0], "o", color="#f39c12", markersize=20,
            markeredgecolor="black", markeredgewidth=2)

    ax.set_title(f"CliffWalking | Step: {step} | Reward: {reward:.0f}")


def save_gif(states, rewards, path, fps=4):
    """Save episode as animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    fig, ax = plt.subplots(figsize=(10, 4))
    path_history = []
    cumulative_reward = 0

    for i, state in enumerate(states):
        pos = state_to_pos(state)
        path_history.append(pos)
        if i > 0:
            cumulative_reward += rewards[i-1]

        draw_frame(ax, pos, path_history, i, cumulative_reward)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
        frames.append(img.convert("RGB"))

    plt.close(fig)

    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000/fps), loop=0)
    return path
