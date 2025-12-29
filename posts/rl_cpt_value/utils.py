"""Visualization utilities for CliffWalking."""

from PIL import Image
from pathlib import Path


def save_gif(frames, path, fps=4):
    """Save frames (numpy arrays) as animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000/fps), loop=0)
    return path
