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
    CPTWeightingFunction,
    PerPathSlidingWindowCPT,
    SlidingWindowCPT,
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
    """REINFORCE agent with Monte Carlo returns and EMA baseline."""

    trainable = True

    def __init__(self, env, lr: float = 1e-3, gamma: float = 0.99, baseline_decay: float = 0.99, baseline_type: str = "ema", **kwargs):
        """Initialize REINFORCE agent.

        Args:
            env: Gymnasium environment with discrete observation/action spaces
            lr: Learning rate for Adam optimizer
            gamma: Discount factor
            baseline_decay: EMA decay for baseline (variance reduction)
            baseline_type: Baseline type ("ema", "min", "max", "zero")
        """
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.policy = PolicyNetwork(self.n_states, self.n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # Baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.baseline_type = baseline_type
        self.best_return = -float("inf")
        self.worst_return = float("inf")
        # Episode storage
        self.log_probs = []
        self.entropies = []
        self.rewards = []

    def act(self, state) -> int:
        """Sample action from policy and store log probability and entropy."""
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        return action.item()

    def _transform_returns(self, returns: list[float]) -> torch.Tensor:
        """Transform returns before policy gradient. Override for custom transforms."""
        return torch.tensor([G - self.baseline for G in returns])

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted Monte Carlo returns from rewards."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def learn(self, env, timesteps: int = 300000, batch_size: int = 8, entropy_coef: float = 0.5, entropy_coef_final: float = 0.01, max_grad_norm: float = 1.0):
        """Train using REINFORCE with full Monte Carlo returns.

        Args:
            env: Gymnasium environment to train on
            timesteps: Total environment steps to train for
            batch_size: Number of episodes to average gradients over
            entropy_coef: Initial entropy coefficient (exploration)
            entropy_coef_final: Final entropy coefficient after annealing
            max_grad_norm: Maximum gradient norm for clipping (None to disable)

        Returns:
            dict with 'episode_rewards' and 'batch_losses' history
        """
        total_steps = 0
        episode = 0
        recent_rewards = []

        # History tracking
        episode_rewards_history = []
        batch_losses_history = []

        while total_steps < timesteps:
            batch_losses = []

            for _ in range(batch_size):
                state, _ = env.reset()
                self.log_probs = []
                self.entropies = []
                self.rewards = []
                done = False

                # Collect full episode (Monte Carlo)
                while not done:
                    action = self.act(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    self.rewards.append(reward)
                    done = terminated or truncated
                    total_steps += 1

                # Compute Monte Carlo returns
                returns = self._compute_returns(self.rewards)

                # Update baseline
                episode_return = returns[0]
                if self.baseline_type == "ema":
                    self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * episode_return
                elif self.baseline_type == "min":
                    self.worst_return = min(self.worst_return, episode_return)
                    self.baseline = self.worst_return
                elif self.baseline_type == "max":
                    self.best_return = max(self.best_return, episode_return)
                    self.baseline = self.best_return
                elif self.baseline_type == "zero":
                    pass

                # Transform returns (baseline subtraction handled inside _transform_returns)
                returns = self._transform_returns(returns)

                # Policy gradient: L = -sum(log_prob * G) - entropy_coef * sum(entropy)
                policy_loss = torch.tensor(0.0)
                entropy_loss = torch.tensor(0.0)
                for log_prob, entropy, G in zip(self.log_probs, self.entropies, returns):
                    policy_loss = policy_loss - log_prob * G
                    entropy_loss = entropy_loss + entropy

                # Compute annealed entropy coefficient
                progress = total_steps / timesteps
                current_entropy_coef = entropy_coef + (entropy_coef_final - entropy_coef) * progress

                loss = policy_loss - current_entropy_coef * entropy_loss
                batch_losses.append(loss)

                # Logging
                ep_reward = sum(self.rewards)
                recent_rewards.append(ep_reward)
                episode_rewards_history.append(ep_reward)
                episode += 1
                if episode % 100 == 0:
                    avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
                    print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")

            # Average batch losses and update with optional gradient clipping
            avg_loss = sum(batch_losses) / batch_size
            batch_losses_history.append(avg_loss.item())
            self.optimizer.zero_grad()
            avg_loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()

        return {
            'episode_rewards': episode_rewards_history,
            'batch_losses': batch_losses_history,
        }


class CPTREINFORCEAgent(REINFORCEAgent):
    """REINFORCE agent with CPT value function and probability weighting.

    Applies Cumulative Prospect Theory (Kahneman & Tversky, 1992) at two levels:
    1. Per-step: CPT value function v(G_t) for within-episode credit assignment
    2. Per-episode: CPT decision weights π_i for across-episode probability distortion

    The decision weights are computed from the empirical return distribution
    using a sliding window, and applied as stop-gradient scalars.

    CPT Parameters:
        alpha: Diminishing sensitivity for gains (default: 0.88)
        beta: Diminishing sensitivity for losses (default: 0.88)
        lambda_: Loss aversion coefficient (default: 2.25)
        reference_point: Gain/loss boundary (default: 0.0)
        use_probability_weighting: Enable CPT probability weighting (default: True)
        w_plus_gamma: Weighting function parameter for gains (default: 0.61)
        w_minus_gamma: Weighting function parameter for losses (default: 0.69)
        sliding_window_size: Number of batches in sliding window (default: 5)
        sliding_window_decay: Exponential decay for old batches (default: 0.8)
    """

    def __init__(
        self,
        env,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
        use_probability_weighting: bool = True,
        w_plus_gamma: float = 0.61,
        w_minus_gamma: float = 0.69,
        sliding_window_size: int = 5,
        sliding_window_decay: float = 0.8,
        lr: float = 1e-3,
        gamma: float = 0.99,
        baseline_type: str = "ema",
        env_config: dict = None,
        **kwargs,
    ):
        super().__init__(env, lr=lr, gamma=gamma, baseline_type=baseline_type)
        self.cpt_value = CPTValueFunction(alpha, beta, lambda_, reference_point)
        self.use_probability_weighting = use_probability_weighting
        if use_probability_weighting:
            weighting_func = CPTWeightingFunction(w_plus_gamma, w_minus_gamma)
            if env_config is not None:
                self.sliding_window = PerPathSlidingWindowCPT(
                    weighting_func, env_config, gamma,
                    max_batches=sliding_window_size,
                    decay=sliding_window_decay,
                    reference_point=reference_point,
                )
            else:
                self.sliding_window = SlidingWindowCPT(
                    weighting_func,
                    max_batches=sliding_window_size,
                    decay=sliding_window_decay,
                    reference_point=reference_point,
                )

    def _transform_returns(self, returns: list[float]) -> torch.Tensor:
        """Apply CPT value function to each per-step return.

        This preserves temporal credit assignment while applying risk-sensitive
        transformation to return-to-go values. Earlier timesteps have larger G_t
        (more future reward to come), so they receive proportionally more credit.
        """
        return torch.tensor([self.cpt_value(G - self.baseline) for G in returns])

    def learn(self, env, timesteps: int = 300000, batch_size: int = 8, entropy_coef: float = 0.5, entropy_coef_final: float = 0.01, max_grad_norm: float = 1.0):
        """Train using REINFORCE with CPT value function and probability weighting.

        Extends base REINFORCE with episode-level CPT decision weights that
        scale each episode's contribution to the policy gradient based on
        where its return falls in the empirical distribution.
        """
        total_steps = 0
        episode = 0
        recent_rewards = []

        episode_rewards_history = []
        batch_losses_history = []

        while total_steps < timesteps:
            # Collect batch of episodes
            batch_data = []  # (log_probs, entropies, returns, episode_return)

            for _ in range(batch_size):
                state, _ = env.reset()
                self.log_probs = []
                self.entropies = []
                self.rewards = []
                done = False

                while not done:
                    action = self.act(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    self.rewards.append(reward)
                    done = terminated or truncated
                    total_steps += 1

                returns = self._compute_returns(self.rewards)
                episode_return = returns[0]

                # Update baseline
                if self.baseline_type == "ema":
                    self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * episode_return
                elif self.baseline_type == "min":
                    self.worst_return = min(self.worst_return, episode_return)
                    self.baseline = self.worst_return
                elif self.baseline_type == "max":
                    self.best_return = max(self.best_return, episode_return)
                    self.baseline = self.best_return
                elif self.baseline_type == "zero":
                    pass

                transformed = self._transform_returns(returns)
                batch_data.append((list(self.log_probs), list(self.entropies), transformed, episode_return))

                ep_reward = sum(self.rewards)
                recent_rewards.append(ep_reward)
                episode_rewards_history.append(ep_reward)
                episode += 1
                if episode % 100 == 0:
                    avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
                    print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")

            # Compute CPT decision weights for this batch
            episode_returns = [d[3] for d in batch_data]

            if self.use_probability_weighting:
                self.sliding_window.add_batch(episode_returns)
                decision_weights = self.sliding_window.compute_decision_weights(episode_returns)
            else:
                decision_weights = np.ones(batch_size)

            # Compute batch loss with decision weights
            batch_losses = []
            for ep_idx, (log_probs, entropies, transformed, _) in enumerate(batch_data):
                w_i = decision_weights[ep_idx]  # stop-gradient scalar
                policy_loss = torch.tensor(0.0)
                entropy_loss = torch.tensor(0.0)
                for log_prob, entropy, G in zip(log_probs, entropies, transformed):
                    policy_loss = policy_loss - log_prob * G * w_i
                    entropy_loss = entropy_loss + entropy

                progress = total_steps / timesteps
                current_entropy_coef = entropy_coef + (entropy_coef_final - entropy_coef) * progress
                loss = policy_loss - current_entropy_coef * entropy_loss
                batch_losses.append(loss)

            avg_loss = sum(batch_losses) / batch_size
            batch_losses_history.append(avg_loss.item())
            self.optimizer.zero_grad()
            avg_loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()

        return {
            'episode_rewards': episode_rewards_history,
            'batch_losses': batch_losses_history,
        }


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
        shape = tuple(cfg["env"]["shape"])
        prompt = get_cliffwalking_prompt(shape, cfg["env"]["reward_cliff"], cfg["env"]["reward_step"])
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
