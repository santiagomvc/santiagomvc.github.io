"""Agent classes with Stable Baselines 3."""

import json
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI
from stable_baselines3 import PPO

from utils import (
    CLIFFWALKING_ACTION_TOOL,
    CLIFFWALKING_PROMPT,
    FROZENLAKE_ACTION_TOOL,
    FROZENLAKE_PROMPT,
    CPTRewardWrapper,
    format_cliffwalking_state,
    format_frozenlake_state,
)


class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def act(self, state):
        """Select action given state."""
        pass

    def learn(self, env, timesteps):
        """Train the agent. Override if trainable."""
        pass

    def close(self):
        """Cleanup resources. Override if needed."""
        pass


class RandomAgent(BaseAgent):
    """Random action agent."""

    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def act(self, state):
        return np.random.randint(self.n_actions)


class PPOAgent(BaseAgent):
    """PPO agent using Stable Baselines 3."""

    def __init__(self, env):
        """Initialize with environment instance."""
        self.model = PPO("MlpPolicy", env, ent_coef=0.1, verbose=1)

    def act(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def learn(self, env, timesteps=50000):
        """Train for given timesteps."""
        self.model.learn(total_timesteps=timesteps)


class CPTPPOAgent(BaseAgent):
    """PPO agent with CPT value function reward shaping.

    Uses Cumulative Prospect Theory's S-shaped value function to transform rewards:
    - Loss aversion (λ=2.25): losses hurt more than equivalent gains
    - Diminishing sensitivity (α=β=0.88): extreme values are compressed

    This is Part 1 of CPT implementation (value side only).
    Part 2 will add cumulative probability weighting.

    Reference: Tversky & Kahneman (1992) "Advances in Prospect Theory"
    """

    def __init__(
        self,
        env,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
    ):
        """Initialize CPT-PPO agent.

        Args:
            env: Gymnasium environment
            alpha: Diminishing sensitivity for gains (default: 0.88)
            beta: Diminishing sensitivity for losses (default: 0.88)
            lambda_: Loss aversion coefficient (default: 2.25)
            reference_point: Baseline for gains/losses (default: 0.0)
        """
        self.cpt_params = {
            "alpha": alpha,
            "beta": beta,
            "lambda": lambda_,
            "reference_point": reference_point,
        }
        self.wrapped_env = CPTRewardWrapper(
            env, alpha, beta, lambda_, reference_point
        )
        self.model = PPO("MlpPolicy", self.wrapped_env, ent_coef=0.1, verbose=1)

    def act(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)

    def learn(self, env, timesteps=50000):
        """Train for given timesteps using CPT-transformed rewards."""
        self.model.learn(total_timesteps=timesteps)

    def close(self):
        """Cleanup wrapped environment."""
        self.wrapped_env.close()


class LLMAgent(BaseAgent):
    """LLM-based agent using OpenAI for action selection.

    Uses function calling with ReAct-style reasoning to navigate the environment.
    No training required - uses prompt engineering for decision making.
    """

    def __init__(self, env, model: str = "gpt-5-mini", verbose: bool = False, env_name: str = "CliffWalking-v0"):
        """Initialize LLM agent.

        Args:
            env: Gymnasium environment
            model: OpenAI model name (default: gpt-5-mini)
            verbose: Print reasoning to stdout (default: False)
            env_name: Environment name for prompt/state/tool selection
        """
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model
        self.verbose = verbose
        self.env_name = env_name

    def _get_action_tool(self) -> dict:
        """Get the action tool with correct mapping for current environment."""
        if "FrozenLake" in self.env_name:
            return FROZENLAKE_ACTION_TOOL
        return CLIFFWALKING_ACTION_TOOL

    def _format_state(self, state: int) -> str:
        """Convert state integer to human-readable description."""
        if "FrozenLake" in self.env_name:
            return format_frozenlake_state(state)
        return format_cliffwalking_state(state)

    def _get_prompt(self) -> str:
        """Get system prompt for current environment."""
        if "FrozenLake" in self.env_name:
            return FROZENLAKE_PROMPT
        return CLIFFWALKING_PROMPT

    def act(self, state) -> int:
        """Select action using LLM with function calling."""
        messages = [
            {"role": "system", "content": self._get_prompt()},
            {
                "role": "user",
                "content": f"Current state:\n{self._format_state(state)}\n\nSelect your action.",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[self._get_action_tool()],
            tool_choice={"type": "function", "function": {"name": "select_action"}},
        )

        # Extract action from function call
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        if self.verbose:
            print(f"State {state}: {args['reasoning']}")
            print(f"  -> Action: {args['action']}")

        return args["action"]


# Agent registry for extensibility
AGENTS = {
    "random": RandomAgent,
    "ppo": PPOAgent,
    "cpt-ppo": CPTPPOAgent,
    "llm": LLMAgent,
}


def get_agent(name, env, **kwargs):
    """Factory function to create agent by name.

    Args:
        name: Agent type ("random", "ppo", "cpt-ppo", "llm")
        env: Gymnasium environment
        **kwargs: Additional arguments passed to agent constructor
            For cpt-ppo: alpha, beta, lambda_, reference_point
            For llm: model, verbose, env_name

    Returns:
        Initialized agent instance
    """
    if name not in AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENTS.keys())}")

    if name == "random":
        return AGENTS[name](n_actions=env.action_space.n)
    elif name == "llm":
        return AGENTS[name](env, **kwargs)
    elif name == "cpt-ppo":
        # Filter out env_name which is only for LLM agent
        cpt_kwargs = {k: v for k, v in kwargs.items() if k != "env_name"}
        return AGENTS[name](env, **cpt_kwargs)
    return AGENTS[name](env)
