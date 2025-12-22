"""Agent functions."""


def random_agent(env):
    """Select a random action."""
    return env.action_space.sample()
