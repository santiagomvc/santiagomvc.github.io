"""Agent classes for CliffWalking RL experiments."""

import json
import math
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
    PerStepSlidingWindowCPT,
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


# class RandomAgent(BaseAgent):
#     """Random action agent."""

#     def __init__(self, n_actions=4):
#         self.n_actions = n_actions

#     def act(self, state):
#         return np.random.randint(self.n_actions)


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


class CPTPGAgent(REINFORCEAgent):
    """CPT-PG agent from Lepel & Barakat (2024), arXiv:2410.02605.

    Applies φ̂(R(τ)) to the total episode return using quantile estimation
    across the batch. All timesteps in an episode share the same scalar weight φ̂.

    Policy gradient: ∇J(θ) = E[φ(R(τ)) · Σ_t ∇θ log π(aₜ|hₜ)]
    """

    trainable = True

    def __init__(
        self,
        env,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_: float = 2.25,
        reference_point: float = 0.0,
        w_plus_gamma: float = 0.61,
        w_minus_gamma: float = 0.69,
        lr: float = 1e-3,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(env, lr=lr, gamma=gamma, baseline_type="zero")
        self.cpt_value = CPTValueFunction(alpha, beta, lambda_, reference_point)
        self.weighting_func = CPTWeightingFunction(w_plus_gamma, w_minus_gamma)

    @staticmethod
    def _integrate_survival(target, sorted_values, n, w_prime_func):
        """Compute ∫₀^target w'(Ŝ(z)) dz using empirical survival function.

        Steps through sorted order statistics. For each interval
        [sorted[k-1], sorted[k]), the unconditional survival probability is
        (m-k)/n where m = len(sorted_values), n = total batch size.
        Rectangle contribution = (min(target, sorted[k]) - prev) * w'(survival).
        """
        if target <= 0 or n == 0:
            return 0.0

        m = len(sorted_values)
        integral = 0.0
        prev = 0.0

        for k in range(m):
            z_k = sorted_values[k]
            if z_k <= 0:
                continue
            if prev >= target:
                break
            # Unconditional survival: fraction of ALL n samples with value >= z_k
            survival = (m - k) / n
            upper = min(target, z_k)
            if upper > prev:
                integral += (upper - prev) * w_prime_func(survival)
            prev = z_k

        # Handle remaining interval if target > max(sorted_values)
        if prev < target:
            integral += (target - prev) * w_prime_func(0.0)

        return integral

    def _compute_phi(self, returns_list):
        """Compute φ̂(R(τ)) for each trajectory in the batch.

        φ(v) = ∫₀^{u⁺(v)} w'₊(Ŝ₊(z))dz - ∫₀^{u⁻(v)} w'₋(Ŝ₋(z))dz
        """
        n = len(returns_list)
        # Compute u(R_i) for each trajectory
        u_values = np.array([self.cpt_value(r) for r in returns_list])

        # Separate into gains and losses
        u_plus = np.maximum(u_values, 0.0)
        u_minus = np.maximum(-u_values, 0.0)

        # Sort for empirical survival function
        sorted_gains = np.sort(u_plus[u_plus > 0])
        sorted_losses = np.sort(u_minus[u_minus > 0])
        n_gains = len(sorted_gains)
        n_losses = len(sorted_losses)

        phi_values = np.zeros(n)
        for i in range(n):
            gain_integral = self._integrate_survival(
                u_plus[i], sorted_gains, n, self.weighting_func.w_prime_plus
            )
            loss_integral = self._integrate_survival(
                u_minus[i], sorted_losses, n, self.weighting_func.w_prime_minus
            )
            phi_values[i] = gain_integral - loss_integral

        # Center φ̂ to remove shared constant offset from w'(1.0) singularity.
        # Equivalent to baseline subtraction in policy gradient theorem (no bias).
        phi_values = phi_values - phi_values.mean()

        return phi_values

    def learn(self, env, timesteps: int = 300000, batch_size: int = 8, entropy_coef: float = 0.5, entropy_coef_final: float = 0.01, max_grad_norm: float = 1.0):
        """Train using CPT-PG (Algorithm 1, Lepel & Barakat 2024).

        Collects batch of trajectories, computes φ̂(R(τ)) for each,
        then uses φ̂ as a scalar weight for the full episode's policy gradient.
        """
        total_steps = 0
        episode = 0
        recent_rewards = []

        episode_rewards_history = []
        batch_losses_history = []

        while total_steps < timesteps:
            batch_data = []  # (log_probs, entropies, total_return)

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

                # Compute total discounted return
                returns = self._compute_returns(self.rewards)
                total_return = returns[0]

                batch_data.append((list(self.log_probs), list(self.entropies), total_return))

                ep_reward = sum(self.rewards)
                recent_rewards.append(ep_reward)
                episode_rewards_history.append(ep_reward)
                episode += 1
                if episode % 100 == 0:
                    avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
                    print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")

            # Compute φ̂ for the batch
            total_returns = [d[2] for d in batch_data]
            phi_values = self._compute_phi(total_returns)

            # Compute batch loss
            batch_losses = []
            for ep_idx, (log_probs, entropies, _) in enumerate(batch_data):
                phi_i = phi_values[ep_idx]
                policy_loss = torch.tensor(0.0)
                entropy_loss = torch.tensor(0.0)
                for log_prob, entropy in zip(log_probs, entropies):
                    policy_loss = policy_loss - phi_i * log_prob
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


# class ReturnPredictor(nn.Module):
#     """Per-step return predictor for RUDDER-style φ̂ decomposition.

#     Takes (state_onehot, action_onehot) → scalar contribution to φ̂.
#     Trained so that Σ_t predictor(s_t, a_t) ≈ φ̂(trajectory).
#     """

#     def __init__(self, n_states: int, n_actions: int, hidden: int = 32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_states + n_actions, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 1),
#         )

#     def forward(self, state_onehot: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
#         x = torch.cat([state_onehot, action_onehot], dim=-1)
#         return self.net(x).squeeze(-1)


# class CPTPGRUDDERAgent(CPTPGAgent):
#     """CPT-PG with RUDDER-style learned per-step decomposition.

#     Trains a per-step MLP (ReturnPredictor) to decompose the episode-level φ̂
#     into per-step contributions φ̃_t, with residual correction to enforce
#     return equivalence (Σ_t φ̃_t = φ̂).

#     The RUDDER model learns which (state, action) pairs contribute most to φ̂,
#     enabling focused credit assignment on causally relevant actions.

#     Based on Arjona-Medina et al. (2019), adapted for CPT-PG's Choquet integral.
#     """

#     def __init__(self, env, rudder_lr: float = 1e-3, rudder_hidden: int = 32, **kwargs):
#         super().__init__(env, **kwargs)
#         self.rudder = ReturnPredictor(self.n_states, self.n_actions, rudder_hidden)
#         self.rudder_optimizer = optim.Adam(self.rudder.parameters(), lr=rudder_lr)

#     def learn(self, env, timesteps: int = 300000, batch_size: int = 8, entropy_coef: float = 0.5, entropy_coef_final: float = 0.01, max_grad_norm: float = 1.0):
#         total_steps = 0
#         episode = 0
#         recent_rewards = []

#         episode_rewards_history = []
#         batch_losses_history = []

#         while total_steps < timesteps:
#             batch_data = []  # (log_probs, entropies, total_return, states, actions)

#             for _ in range(batch_size):
#                 state, _ = env.reset()
#                 self.log_probs = []
#                 self.entropies = []
#                 self.rewards = []
#                 states = []
#                 actions = []
#                 done = False

#                 while not done:
#                     states.append(state)
#                     action = self.act(state)
#                     actions.append(action)
#                     state, reward, terminated, truncated, _ = env.step(action)
#                     self.rewards.append(reward)
#                     done = terminated or truncated
#                     total_steps += 1

#                 returns = self._compute_returns(self.rewards)
#                 total_return = returns[0]

#                 batch_data.append((list(self.log_probs), list(self.entropies), total_return, states, actions))

#                 ep_reward = sum(self.rewards)
#                 recent_rewards.append(ep_reward)
#                 episode_rewards_history.append(ep_reward)
#                 episode += 1
#                 if episode % 100 == 0:
#                     avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
#                     print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")

#             # Compute centered φ̂ for the batch
#             total_returns = [d[2] for d in batch_data]
#             phi_values = self._compute_phi(total_returns)

#             # Train RUDDER model: predict per-step contributions summing to φ̂
#             rudder_loss = torch.tensor(0.0)
#             for ep_idx, (_, _, _, states, actions) in enumerate(batch_data):
#                 phi_target = phi_values[ep_idx]
#                 pred_sum = torch.tensor(0.0)
#                 for s, a in zip(states, actions):
#                     s_onehot = torch.zeros(self.n_states)
#                     s_onehot[s] = 1.0
#                     a_onehot = torch.zeros(self.n_actions)
#                     a_onehot[a] = 1.0
#                     pred_sum = pred_sum + self.rudder(s_onehot, a_onehot)
#                 rudder_loss = rudder_loss + (pred_sum - phi_target) ** 2

#             rudder_loss = rudder_loss / batch_size
#             self.rudder_optimizer.zero_grad()
#             rudder_loss.backward()
#             self.rudder_optimizer.step()

#             # Decompose φ̂ to per-step using trained RUDDER model
#             batch_losses = []
#             for ep_idx, (log_probs, entropies, _, states, actions) in enumerate(batch_data):
#                 phi_i = phi_values[ep_idx]

#                 # Get per-step predictions (detached — no gradient through RUDDER for policy)
#                 phi_tilde = []
#                 for s, a in zip(states, actions):
#                     s_onehot = torch.zeros(self.n_states)
#                     s_onehot[s] = 1.0
#                     a_onehot = torch.zeros(self.n_actions)
#                     a_onehot[a] = 1.0
#                     phi_tilde.append(self.rudder(s_onehot, a_onehot).detach().item())

#                 # Enforce return equivalence: distribute residual uniformly
#                 residual = phi_i - sum(phi_tilde)
#                 T = len(phi_tilde)
#                 phi_tilde = [p + residual / T for p in phi_tilde]

#                 policy_loss = torch.tensor(0.0)
#                 entropy_loss = torch.tensor(0.0)
#                 for log_prob, entropy, phi_t in zip(log_probs, entropies, phi_tilde):
#                     policy_loss = policy_loss - phi_t * log_prob
#                     entropy_loss = entropy_loss + entropy

#                 progress = total_steps / timesteps
#                 current_entropy_coef = entropy_coef + (entropy_coef_final - entropy_coef) * progress
#                 loss = policy_loss - current_entropy_coef * entropy_loss
#                 batch_losses.append(loss)

#             avg_loss = sum(batch_losses) / batch_size
#             batch_losses_history.append(avg_loss.item())
#             self.optimizer.zero_grad()
#             avg_loss.backward()
#             if max_grad_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=max_grad_norm)
#             self.optimizer.step()

#         return {
#             'episode_rewards': episode_rewards_history,
#             'batch_losses': batch_losses_history,
#         }


# class PerStepCPTAgent(REINFORCEAgent):
#     """REINFORCE agent with mathematically sound per-step CPT.

#     Applies CPT probability weighting at each timestep on the distribution of
#     G_t across episodes, with a separated baseline that reduces variance without
#     biasing the objective.

#     1. Separated baseline: v(G_t) - v(b_t) instead of v(G_t - b_t)
#        Subtracting v(b_t) after v() preserves the unbiased gradient.
#     2. Per-step decision weights π_{i,t} from PerStepSlidingWindowCPT
#        Treats {G_t^(1), ..., G_t^(N)} as a prospect at each t.
#     3. Per-timestep baselines b_t (one EMA baseline per timestep position)

#     Policy loss: L_i = -Σ_t π_{i,t} · [v(G_t) - v(b_t)] · log π(a_t|s_t) - c(k)·H
#     """

#     trainable = True

#     def __init__(
#         self,
#         env,
#         alpha: float = 0.88,
#         beta: float = 0.88,
#         lambda_: float = 2.25,
#         reference_point: float = 0.0,
#         w_plus_gamma: float = 0.61,
#         w_minus_gamma: float = 0.69,
#         sliding_window_size: int = 5,
#         sliding_window_decay: float = 0.8,
#         lr: float = 1e-3,
#         gamma: float = 0.99,
#         baseline_decay: float = 0.99,
#         max_is_ratio: float = 5.0,
#         **kwargs,
#     ):
#         super().__init__(env, lr=lr, gamma=gamma, baseline_decay=baseline_decay)
#         self.cpt_value = CPTValueFunction(alpha, beta, lambda_, reference_point)
#         self.max_is_ratio = max_is_ratio
#         self.baselines = {}  # per-timestep EMA baselines: {t: float}
#         weighting_func = CPTWeightingFunction(w_plus_gamma, w_minus_gamma)
#         self.sliding_window = PerStepSlidingWindowCPT(
#             weighting_func,
#             max_batches=sliding_window_size,
#             decay=sliding_window_decay,
#             reference_point=reference_point,
#         )

#     def learn(self, env, timesteps: int = 300000, batch_size: int = 8, entropy_coef: float = 0.5, entropy_coef_final: float = 0.01, max_grad_norm: float = 1.0):
#         """Train using REINFORCE with per-step CPT probability weighting.

#         Collects batches of episodes, computes per-step decision weights π_{i,t}
#         from the distribution of G_t at each timestep, then uses separated
#         baseline v(G_t) - v(b_t) for the advantage.
#         """
#         total_steps = 0
#         episode = 0
#         recent_rewards = []

#         episode_rewards_history = []
#         batch_losses_history = []

#         while total_steps < timesteps:
#             # Collect batch of episodes
#             batch_data = []  # (log_probs, entropies, per_step_returns, metadata)

#             for _ in range(batch_size):
#                 state, _ = env.reset()
#                 self.log_probs = []
#                 self.entropies = []
#                 self.rewards = []
#                 states = []
#                 actions = []
#                 done = False

#                 while not done:
#                     states.append(state)
#                     action = self.act(state)
#                     actions.append(action)
#                     state, reward, terminated, truncated, _ = env.step(action)
#                     self.rewards.append(reward)
#                     done = terminated or truncated
#                     total_steps += 1

#                 # Compute per-step Monte Carlo returns [G_0, G_1, ..., G_T]
#                 returns = self._compute_returns(self.rewards)
#                 old_log_prob = sum(lp.detach().item() for lp in self.log_probs)

#                 batch_data.append((list(self.log_probs), list(self.entropies), returns, (states, actions, old_log_prob)))

#                 ep_reward = sum(self.rewards)
#                 recent_rewards.append(ep_reward)
#                 episode_rewards_history.append(ep_reward)
#                 episode += 1
#                 if episode % 100 == 0:
#                     avg = sum(recent_rewards[-100:]) / min(100, len(recent_rewards))
#                     print(f"Ep {episode} | Steps {total_steps} | Reward {ep_reward:.1f} | Avg100 {avg:.1f}")

#             # Gather per-step returns for all episodes
#             all_per_step_returns = [d[2] for d in batch_data]

#             # Compute per-step CPT decision weights with importance sampling
#             all_metadata = [d[3] for d in batch_data]
#             self.sliding_window.add_batch(all_per_step_returns, all_metadata)

#             def _is_ratio_fn(meta):
#                 states, actions, old_log_prob = meta
#                 with torch.no_grad():
#                     new_log_prob = sum(
#                         torch.log(self.policy(s)[a]).item()
#                         for s, a in zip(states, actions)
#                     )
#                 log_ratio = min(new_log_prob - old_log_prob, math.log(self.max_is_ratio))
#                 return math.exp(log_ratio)

#             per_step_weights = self.sliding_window.compute_decision_weights(
#                 all_per_step_returns, _is_ratio_fn
#             )

#             # Compute batch loss with per-step weights and separated baseline
#             batch_losses = []
#             # Accumulate G_t values per timestep for baseline updates
#             timestep_returns = {}  # {t: [G_t values across episodes]}

#             for ep_idx, (log_probs, entropies, returns, _meta) in enumerate(batch_data):
#                 ep_weights = per_step_weights[ep_idx]

#                 policy_loss = torch.tensor(0.0)
#                 entropy_loss = torch.tensor(0.0)

#                 for t, (log_prob, entropy, G_t) in enumerate(zip(log_probs, entropies, returns)):
#                     # Per-timestep baseline
#                     b_t = self.baselines.get(t, 0.0)

#                     # Separated baseline: v(G_t) - v(b_t)
#                     advantage = self.cpt_value(G_t) - self.cpt_value(b_t)

#                     # Per-step decision weight
#                     w_it = ep_weights[t] if t < len(ep_weights) else 1.0

#                     policy_loss = policy_loss - log_prob * w_it * advantage
#                     entropy_loss = entropy_loss + entropy

#                     # Track returns for baseline update
#                     if t not in timestep_returns:
#                         timestep_returns[t] = []
#                     timestep_returns[t].append(G_t)

#                 progress = total_steps / timesteps
#                 current_entropy_coef = entropy_coef + (entropy_coef_final - entropy_coef) * progress
#                 loss = policy_loss - current_entropy_coef * entropy_loss
#                 batch_losses.append(loss)

#             # Update per-timestep baselines
#             for t, g_values in timestep_returns.items():
#                 mean_g = sum(g_values) / len(g_values)
#                 old_b = self.baselines.get(t, 0.0)
#                 self.baselines[t] = self.baseline_decay * old_b + (1 - self.baseline_decay) * mean_g

#             avg_loss = sum(batch_losses) / batch_size
#             batch_losses_history.append(avg_loss.item())
#             self.optimizer.zero_grad()
#             avg_loss.backward()
#             if max_grad_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=max_grad_norm)
#             self.optimizer.step()

#         return {
#             'episode_rewards': episode_rewards_history,
#             'batch_losses': batch_losses_history,
#         }


# class LLMAgent(BaseAgent):
#     """LLM-based agent using OpenAI for action selection.

#     Uses function calling with ReAct-style reasoning to navigate the environment.
#     No training required - uses prompt engineering for decision making.
#     """

#     def __init__(self, env, model: str = "gpt-5-mini", verbose: bool = False):
#         """Initialize LLM agent.

#         Args:
#             env: Gymnasium environment
#             model: OpenAI model name (default: gpt-5-mini)
#             verbose: Print reasoning to stdout (default: False)
#         """
#         self.client = OpenAI()  # Uses OPENAI_API_KEY env var
#         self.model = model
#         self.verbose = verbose

#     def act(self, state) -> int:
#         """Select action using LLM with function calling."""
#         cfg = load_config()
#         shape = tuple(cfg["env"]["shape"])
#         prompt = get_cliffwalking_prompt(shape, cfg["env"]["reward_cliff"], cfg["env"]["reward_step"])
#         messages = [
#             {"role": "system", "content": prompt},
#             {
#                 "role": "user",
#                 "content": f"Current state:\n{format_cliffwalking_state(state, shape)}\n\nSelect your action.",
#             },
#         ]

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             tools=[CLIFFWALKING_ACTION_TOOL],
#             tool_choice={"type": "function", "function": {"name": "select_action"}},
#         )

#         # Extract action from function call
#         tool_call = response.choices[0].message.tool_calls[0]
#         args = json.loads(tool_call.function.arguments)

#         if self.verbose:
#             print(f"State {state}: {args['reasoning']}")
#             print(f"  -> Action: {args['action']}")

#         return args["action"]


# Agent registry for extensibility
AGENTS = {
    # "random": RandomAgent,
    "reinforce": REINFORCEAgent,
    # "per-step-cpt": PerStepCPTAgent,
    "cpt-pg": CPTPGAgent,
    # "cpt-pg-rudder": CPTPGRUDDERAgent,
    # "llm": LLMAgent,
}


def get_agent(name, env, **kwargs):
    """Factory function to create agent by name.

    Args:
        name: Agent type ("random", "reinforce", "per-step-cpt", "llm")
        env: Gymnasium environment
        **kwargs: Additional arguments passed to agent constructor

    Returns:
        Initialized agent instance
    """
    if name not in AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENTS.keys())}")

    if name == "random":
        return AGENTS[name](n_actions=env.action_space.n)
    elif name in ("llm", "reinforce", "per-step-cpt", "cpt-pg", "cpt-pg-rudder"):
        return AGENTS[name](env, **kwargs)
