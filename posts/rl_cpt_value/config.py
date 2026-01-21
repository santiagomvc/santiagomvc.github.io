"""Configuration for CliffWalking experiments."""

SHAPE = (5, 5)
STOCHASTICITY = "windy"  # "slippery", "windy", or None
REWARD_CLIFF = -50
REWARD_STEP = -1
WIND_PROB = 0.2
TIMESTEPS = 300000
N_EVAL_EPISODES = 5
