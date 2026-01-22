"""Agent classes for CliffWalking RL experiments."""

import json
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI
from torch.distributions import Categorical

from utils import (
    load_config,
    CLIFFWALKING_ACTION_TOOL,
    get_cliffwalking_prompt,
    CPTValueFunction,
    format_cliffwalking_state,
)


class BaseAgent(ABC):
    """Base class for all agents."""

    trainable = False

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


class PolicyNetwork(nn.Module):
    """Simple MLP policy network for discrete action spaces."""

    def __init__(self, n_states: int = 25, n_actions: int = 4, hidden: int = 64):
        super().__init__()
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: int) -> torch.Tensor:
        """Convert state int to one-hot and return action probabilities."""
        x = torch.zeros(self.n_states)
        x[state] = 1.0
        return torch.softmax(self.net(x), dim=-1)


class REINFORCEAgent(BaseAgent):
    """Vanilla REINFORCE agent with Monte Carlo returns (no baseline)."""

    trainable = True

    def __init__(self, env, lr: float = 1e-3, gamma: float = 0.99):
        """Initialize REINFORCE agent.

        Args:
            env: Gymnasium environment with discrete observation/action spaces
            lr: Learning rate for Adam optimizer
            gamma: Discount factor
        """
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.policy = PolicyNetwork(self.n_states, self.n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # Episode storage
        self.log_probs = []
        self.rewards = []

    def act(self, state) -> int:
        """Sample action from policy and store log probability."""
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def _transform_returns(self, returns: list[float]) -> torch.Tensor:
        """Transform returns before policy gradient. Override for custom transforms."""
        return torch.tensor(returns)

    def learn(self, env, timesteps: int = 300000):
        """Train using REINFORCE with full Monte Carlo returns.

        Args:
            env: Gymnasium environment to train on
            timesteps: Total environment steps to train for
        """
        total_steps = 0
        episode = 0
        recent_rewards = []

        while total_steps < timesteps:
            state, _ = env.reset()
            self.log_probs = []
            self.rewards = []
            done = False

            # Collect full episode (Monte Carlo)
            while not done:
                action = self.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                self.rewards.append(reward)
                done = terminated or truncated
                total_steps += 1

            # Compute Monte Carlo returns (no baseline)
            returns = []
            G = 0
            for r in reversed(self.rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = self._transform_returns(returns)

            # Policy gradient: L = -sum(log_prob * G)
            loss = torch.tensor(0.0)
            for log_prob, G in zip(self.log_probs, returns):
                loss = loss - log_prob * G

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            ep_reward = sum(self.rewards)
            recent_rewards.append(ep_reward)
            episode += 1
            if episode % 100 == 0:
                avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
                print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")


class CPTREINFORCEAgent(REINFORCEAgent):
    """REINFORCE agent with CPT applied to episode returns."""

    def __init__(
        self,
        env,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        super().__init__(env, lr=lr, gamma=gamma)
        self.cpt_value = CPTValueFunction(alpha, beta, lambda_, reference_point)

    def _transform_returns(self, returns: list[float]) -> torch.Tensor:
        """Apply CPT value function to each return."""
        return torch.tensor([self.cpt_value(G) for G in returns])


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
        cfg = load_config()
        shape = tuple(cfg["shape"])
        prompt = get_cliffwalking_prompt(shape, cfg["reward_cliff"], cfg["reward_step"])
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Current state:\n{format_cliffwalking_state(state, shape)}\n\nSelect your action.",
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
    "reinforce": REINFORCEAgent,
    "cpt-reinforce": CPTREINFORCEAgent,
    "llm": LLMAgent,
}


def get_agent(name, env, **kwargs):
    """Factory function to create agent by name.

    Args:
        name: Agent type ("random", "reinforce", "cpt-reinforce", "llm")
        env: Gymnasium environment
        **kwargs: Additional arguments passed to agent constructor
            For reinforce: lr, gamma
            For cpt-reinforce: alpha, beta, lambda_, reference_point, lr, gamma
            For llm: model, verbose

    Returns:
        Initialized agent instance
    """
    if name not in AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENTS.keys())}")

    if name == "random":
        return AGENTS[name](n_actions=env.action_space.n)
    elif name in ("llm", "reinforce", "cpt-reinforce"):
        return AGENTS[name](env, **kwargs)
    return AGENTS[name](env)
