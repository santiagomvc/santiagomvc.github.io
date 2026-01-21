"""Agent classes with Stable Baselines 3."""

import json
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI
from stable_baselines3 import PPO

import config
from utils import (
    CLIFFWALKING_ACTION_TOOL,
    get_cliffwalking_prompt,
    CPTRewardWrapper,
    format_cliffwalking_state,
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
        print(f"[CPT-PPO] Initializing with CPT params: α={alpha}, β={beta}, λ={lambda_}, ref={reference_point}")
        self.cpt_params = {
            "alpha": alpha,
            "beta": beta,
            "lambda": lambda_,
            "reference_point": reference_point,
        }
        self.wrapped_env = CPTRewardWrapper(
            env, alpha, beta, lambda_, reference_point
        )
        print("[CPT-PPO] Environment wrapped with CPT reward transformation")
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

    def __init__(self, env, model: str = "gpt-5-mini", verbose: bool = False):
        """Initialize LLM agent.

        Args:
            env: Gymnasium environment
            model: OpenAI model name (default: gpt-5-mini)
            verbose: Print reasoning to stdout (default: False)
        """
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model
        self.verbose = verbose

    def act(self, state) -> int:
        """Select action using LLM with function calling."""
        prompt = get_cliffwalking_prompt(config.SHAPE, config.REWARD_CLIFF, config.REWARD_STEP)
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Current state:\n{format_cliffwalking_state(state, config.SHAPE)}\n\nSelect your action.",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[CLIFFWALKING_ACTION_TOOL],
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
            For llm: model, verbose

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
        return AGENTS[name](env, **kwargs)
    return AGENTS[name](env)
