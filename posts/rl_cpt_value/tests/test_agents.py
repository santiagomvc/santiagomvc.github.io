"""Tests for REINFORCE and CPT-REINFORCE agents."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    REINFORCEAgent,
    CPTREINFORCEAgent,
    AGENTS,
    get_agent,
)
from utils import CPTValueFunction


class TestMonteCarloReturns:
    """Tests for Monte Carlo return computation."""

    def test_return_formula_single_step(self):
        """G_T = r_T for the last step."""
        rewards = [10.0]
        gamma = 0.99

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        assert len(returns) == 1
        assert returns[0] == 10.0

    def test_return_formula_multiple_steps(self):
        """G_t = r_t + gamma * G_{t+1}."""
        rewards = [-1.0, -1.0, -1.0, 10.0]
        gamma = 0.99

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Manual calculation:
        # G_3 = 10.0
        # G_2 = -1.0 + 0.99 * 10.0 = 8.9
        # G_1 = -1.0 + 0.99 * 8.9 = 7.811
        # G_0 = -1.0 + 0.99 * 7.811 = 6.73289
        assert abs(returns[3] - 10.0) < 1e-10
        assert abs(returns[2] - 8.9) < 1e-10
        assert abs(returns[1] - 7.811) < 1e-10
        assert abs(returns[0] - 6.73289) < 1e-5

    def test_gamma_zero_returns_immediate_reward(self):
        """With gamma=0, G_t = r_t."""
        rewards = [-1.0, -2.0, -3.0]
        gamma = 0.0

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for ret, rew in zip(returns, rewards):
            assert ret == rew

    def test_gamma_one_sums_all_rewards(self):
        """With gamma=1, G_0 = sum of all rewards."""
        rewards = [-1.0, -1.0, -1.0, -1.0]
        gamma = 1.0

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        assert returns[0] == sum(rewards)

    def test_negative_rewards_produce_negative_returns(self):
        """All negative rewards should produce negative returns."""
        rewards = [-1.0, -1.0, -50.0]
        gamma = 0.99

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for ret in returns:
            assert ret < 0


class TestREINFORCEAgent:
    """Tests for vanilla REINFORCE agent implementation."""

    def test_initialization(self, small_env, torch_seed):
        """Agent should initialize with correct parameters."""
        agent = REINFORCEAgent(small_env, lr=1e-3, gamma=0.99)

        assert agent.n_states == small_env.observation_space.n
        assert agent.n_actions == small_env.action_space.n
        assert agent.gamma == 0.99
        assert len(agent.log_probs) == 0
        assert len(agent.rewards) == 0

    def test_act_returns_valid_action(self, reinforce_agent, small_env):
        """act() should return valid action in action space."""
        state, _ = small_env.reset()
        action = reinforce_agent.act(state)

        assert 0 <= action < small_env.action_space.n
        assert isinstance(action, int)

    def test_act_stores_log_prob(self, reinforce_agent, small_env):
        """act() should store log probability."""
        state, _ = small_env.reset()
        initial_len = len(reinforce_agent.log_probs)

        reinforce_agent.act(state)

        assert len(reinforce_agent.log_probs) == initial_len + 1
        assert isinstance(reinforce_agent.log_probs[-1], torch.Tensor)

    def test_log_probs_are_negative(self, reinforce_agent, small_env):
        """Log probabilities should be <= 0."""
        state, _ = small_env.reset()
        reinforce_agent.act(state)

        assert reinforce_agent.log_probs[-1].item() <= 0

    def test_transform_returns_identity(self, reinforce_agent):
        """Base REINFORCE should not transform returns."""
        returns = [-10.0, -5.0, 0.0, 5.0, 10.0]
        transformed = reinforce_agent._transform_returns(returns)

        assert isinstance(transformed, torch.Tensor)
        for orig, trans in zip(returns, transformed.tolist()):
            assert orig == trans

    def test_policy_produces_valid_distribution(self, reinforce_agent, small_env):
        """Policy network should produce valid probability distribution."""
        state, _ = small_env.reset()
        probs = reinforce_agent.policy(state)

        assert probs.shape == (small_env.action_space.n,)
        assert abs(probs.sum().item() - 1.0) < 1e-6
        assert all(p >= 0 for p in probs.tolist())


class TestCPTValueFunction:
    """Tests for CPT value function correctness."""

    def test_gains_formula(self):
        """v(x) = x^alpha for x >= 0."""
        alpha = 0.88
        v = CPTValueFunction(alpha=alpha)

        test_values = [0, 1, 10, 50, 100]
        for x in test_values:
            if x == 0:
                assert v(x) == 0.0
            else:
                expected = x ** alpha
                assert abs(v(x) - expected) < 1e-10

    def test_losses_formula(self):
        """v(x) = -lambda * (-x)^beta for x < 0."""
        beta = 0.88
        lambda_ = 2.25
        v = CPTValueFunction(beta=beta, lambda_=lambda_)

        test_values = [-1, -10, -50, -100]
        for x in test_values:
            expected = -lambda_ * ((-x) ** beta)
            assert abs(v(x) - expected) < 1e-10

    def test_loss_aversion(self):
        """|v(-x)| > v(x) for same magnitude when lambda > 1."""
        v = CPTValueFunction(alpha=0.88, beta=0.88, lambda_=2.25)

        test_values = [1, 10, 50, 100]
        for x in test_values:
            gain = v(x)
            loss_magnitude = abs(v(-x))
            assert loss_magnitude > gain, \
                f"|v({-x})|={loss_magnitude} should be > v({x})={gain}"

    def test_reference_point_shifts_boundary(self):
        """Reference point should shift gain/loss classification."""
        v_ref0 = CPTValueFunction(reference_point=0.0)
        v_ref_neg10 = CPTValueFunction(reference_point=-10.0)

        # With reference=0, x=-5 is a loss
        assert v_ref0(-5) < 0

        # With reference=-10, x=-5 is a gain (better than -10)
        assert v_ref_neg10(-5) > 0

    def test_zero_at_reference_point(self):
        """v(reference_point) should equal 0."""
        for ref in [0.0, -10.0, 5.0]:
            v = CPTValueFunction(reference_point=ref)
            assert v(ref) == 0.0


class TestCPTREINFORCEAgent:
    """Tests for CPT-REINFORCE agent implementation."""

    def test_inherits_from_reinforce(self, small_env, torch_seed):
        """CPTREINFORCEAgent should inherit from REINFORCEAgent."""
        agent = CPTREINFORCEAgent(small_env)
        assert isinstance(agent, REINFORCEAgent)

    def test_has_cpt_value_function(self, cpt_reinforce_agent):
        """CPT agent should have a CPT value function."""
        assert hasattr(cpt_reinforce_agent, 'cpt_value')
        assert isinstance(cpt_reinforce_agent.cpt_value, CPTValueFunction)

    def test_cpt_parameters_stored(self, small_env, torch_seed):
        """CPT parameters should be correctly stored."""
        agent = CPTREINFORCEAgent(
            small_env,
            alpha=0.75,
            beta=0.85,
            lambda_=3.0,
            reference_point=-5.0,
        )

        assert agent.cpt_value.alpha == 0.75
        assert agent.cpt_value.beta == 0.85
        assert agent.cpt_value.lambda_ == 3.0
        assert agent.cpt_value.reference_point == -5.0

    def test_cpt_applied_to_episode_return_broadcast(self, small_env, torch_seed):
        """CPT is applied to G_0 (episode return) and broadcast to all timesteps.

        This is the critical test verifying that:
        1. Monte Carlo returns are computed first: G_t = r_t + gamma*G_{t+1}
        2. CPT is applied only to G_0 (episode return): v(G_0)
        3. This single value is broadcast to all timesteps

        This avoids gradient accumulation bias from episode length variation.
        """
        agent = CPTREINFORCEAgent(small_env, alpha=0.88, beta=0.88, lambda_=2.25)
        gamma = agent.gamma

        # Episode rewards (step-by-step)
        step_rewards = [-1.0, -1.0, -1.0, -50.0]

        # Step 1: Compute Monte Carlo returns first
        returns = []
        G = 0
        for r in reversed(step_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Step 2: Agent applies CPT to episode return and broadcasts
        transformed = agent._transform_returns(returns)

        # Expected: CPT(G_0) broadcast to all timesteps
        cpt_v = agent.cpt_value
        expected_cpt_episode = cpt_v(returns[0])  # CPT of episode return G_0

        # All transformed values should equal CPT(G_0)
        for i, trans in enumerate(transformed.tolist()):
            assert abs(trans - expected_cpt_episode) < 1e-5, \
                f"Step {i}: transformed={trans} should equal CPT(G_0)={expected_cpt_episode}"

    def test_transform_returns_applies_cpt(self, cpt_reinforce_agent):
        """_transform_returns should apply CPT to G_0 and broadcast."""
        returns = [-10.0, -5.0, 0.0, 5.0, 10.0]
        cpt_v = cpt_reinforce_agent.cpt_value

        transformed = cpt_reinforce_agent._transform_returns(returns)

        # CPT is applied to G_0 only and broadcast to all timesteps
        expected_cpt_episode = cpt_v(returns[0])
        for trans in transformed.tolist():
            assert abs(trans - expected_cpt_episode) < 1e-6

    def test_negative_returns_become_more_negative_with_loss_aversion(self, small_env, torch_seed):
        """With lambda > 1, negative episode return should become more negative."""
        agent = CPTREINFORCEAgent(small_env, alpha=0.88, beta=0.88, lambda_=2.25)
        episode_return = -50.0
        returns = [episode_return, -30.0, -10.0]  # G_0 is episode return

        transformed = agent._transform_returns(returns)

        # CPT with loss aversion should amplify losses (applied to G_0, broadcast)
        # |v(x)| > |x| for losses when lambda > 1 and beta < 1
        cpt_episode = transformed[0].item()
        assert cpt_episode < episode_return, f"v({episode_return})={cpt_episode} should be more negative"
        # All values should be the same (broadcast)
        assert all(t == cpt_episode for t in transformed.tolist())


class TestREINFORCELearning:
    """Tests for REINFORCE learning behavior."""

    def test_reinforce_improves_over_training(self, env_factory, torch_seed):
        """REINFORCE should show learning progress."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = REINFORCEAgent(env, lr=1e-2, gamma=0.99)

        # Collect rewards before training
        initial_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            initial_rewards.append(episode_reward)
            agent.log_probs = []
            agent.entropies = []
            agent.rewards = []

        # Train
        agent.learn(env, timesteps=5000)

        # Collect rewards after training
        final_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            final_rewards.append(episode_reward)
            agent.log_probs = []
            agent.entropies = []
            agent.rewards = []

        # Performance should improve (less negative reward)
        initial_avg = np.mean(initial_rewards)
        final_avg = np.mean(final_rewards)

        assert final_avg > initial_avg, \
            f"Final avg reward {final_avg} should be > initial {initial_avg}"

    def test_reinforce_converges_on_simple_env(self, env_factory, torch_seed):
        """REINFORCE should find good policy on deterministic 3x3 grid."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = REINFORCEAgent(env, lr=1e-2, gamma=0.99)

        # Optimal path in 3x3: UP, RIGHT, DOWN = 3 steps = -3 reward
        # (Start at bottom-left, goal at bottom-right, cliff in middle)
        optimal_reward = -3.0

        # Use batch_size=1, no entropy, no gradient clipping for this simple test
        agent.learn(env, timesteps=10000, batch_size=1, entropy_coef=0.0, max_grad_norm=None)

        # Evaluate learned policy
        rewards = []
        for _ in range(50):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
            agent.log_probs = []
            agent.entropies = []
            agent.rewards = []

        avg_reward = np.mean(rewards)

        # Should achieve near-optimal performance
        assert avg_reward >= optimal_reward - 2.0, \
            f"Avg reward {avg_reward} should be >= {optimal_reward - 2.0}"


class TestCPTREINFORCELearning:
    """Tests for CPT-REINFORCE learning behavior."""

    def test_cpt_agent_improves_over_training(self, env_factory, torch_seed):
        """CPT agent should show learning progress."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = CPTREINFORCEAgent(env, lr=1e-2, gamma=0.99, alpha=0.88, beta=0.88, lambda_=2.25)

        # Collect rewards before training (using raw environment rewards, not CPT-transformed)
        initial_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            initial_rewards.append(episode_reward)
            agent.log_probs = []
            agent.entropies = []
            agent.rewards = []

        # Train
        agent.learn(env, timesteps=5000)

        # Collect rewards after training
        final_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            final_rewards.append(episode_reward)
            agent.log_probs = []
            agent.entropies = []
            agent.rewards = []

        initial_avg = np.mean(initial_rewards)
        final_avg = np.mean(final_rewards)

        assert final_avg > initial_avg, \
            f"CPT agent should improve: final {final_avg} > initial {initial_avg}"

    def test_cpt_agent_behavior_differs_from_standard(self, env_factory, torch_seed):
        """CPT loss aversion should lead to different learned policies.

        In environments with cliff risk, CPT agents with high loss aversion
        should prefer safer (though possibly longer) paths.
        """
        # Use environment where risk matters
        env = env_factory(shape=(5, 5), wind_prob=0.3, reward_cliff=-100.0, reward_step=-1.0)

        torch.manual_seed(42)
        np.random.seed(42)
        standard_agent = REINFORCEAgent(env, lr=1e-2, gamma=0.99)

        torch.manual_seed(42)
        np.random.seed(42)
        cpt_agent = CPTREINFORCEAgent(env, lr=1e-2, gamma=0.99, alpha=0.88, beta=0.88, lambda_=5.0)

        # Train both agents
        standard_agent.learn(env, timesteps=10000)
        cpt_agent.learn(env, timesteps=10000)

        # Evaluate policies - check action distributions
        test_state = 0  # Top-left corner

        standard_probs = standard_agent.policy(test_state).detach().numpy()
        cpt_probs = cpt_agent.policy(test_state).detach().numpy()

        # Policies should differ - check max absolute difference
        max_diff = np.max(np.abs(standard_probs - cpt_probs))
        assert max_diff > 0.01, \
            f"Policies should differ: max diff {max_diff:.4f}, standard={standard_probs}, cpt={cpt_probs}"

    def test_higher_loss_aversion_transforms_differently(self, env_factory, torch_seed):
        """Higher lambda should transform episode return more aggressively.

        This tests the mechanism rather than stochastic learning outcomes.
        """
        env = env_factory(shape=(5, 5), wind_prob=0.0, reward_cliff=-100.0, reward_step=-1.0)

        low_lambda_agent = CPTREINFORCEAgent(env, lr=1e-2, gamma=0.99, lambda_=1.0)
        high_lambda_agent = CPTREINFORCEAgent(env, lr=1e-2, gamma=0.99, lambda_=5.0)

        # Test on negative episode return (loss) - G_0 is the episode return
        episode_return = -100.0
        returns = [episode_return, -50.0, -20.0, -10.0]  # G_0 is episode return

        low_transformed = low_lambda_agent._transform_returns(returns)
        high_transformed = high_lambda_agent._transform_returns(returns)

        # Both should broadcast CPT(G_0) to all timesteps
        low_cpt = low_transformed[0].item()
        high_cpt = high_transformed[0].item()

        # Both should be negative
        assert low_cpt < 0
        assert high_cpt < 0

        # Higher lambda should produce more negative value (more loss averse)
        assert high_cpt < low_cpt, \
            f"High lambda v({episode_return})={high_cpt} should be < low lambda v({episode_return})={low_cpt}"

        # All values in each tensor should be the same (broadcast)
        assert all(t == low_cpt for t in low_transformed.tolist())
        assert all(t == high_cpt for t in high_transformed.tolist())


class TestAgentRegistry:
    """Tests for agent registry and factory function."""

    def test_agents_dict_contains_expected_agents(self):
        """AGENTS dict should contain all expected agent types."""
        expected_agents = ["random", "reinforce", "cpt-reinforce", "llm"]
        for agent_name in expected_agents:
            assert agent_name in AGENTS, f"'{agent_name}' should be in AGENTS"

    def test_cpt_episode_reinforce_not_in_registry(self):
        """cpt-episode-reinforce should NOT be in AGENTS (consolidated into cpt-reinforce)."""
        assert "cpt-episode-reinforce" not in AGENTS

    def test_get_agent_creates_reinforce(self, small_env):
        """get_agent should create REINFORCEAgent."""
        agent = get_agent("reinforce", small_env, lr=1e-3, gamma=0.99)
        assert isinstance(agent, REINFORCEAgent)

    def test_get_agent_creates_cpt_reinforce(self, small_env):
        """get_agent should create CPTREINFORCEAgent."""
        agent = get_agent("cpt-reinforce", small_env, alpha=0.88, beta=0.88, lambda_=2.25)
        assert isinstance(agent, CPTREINFORCEAgent)

    def test_get_agent_unknown_raises_error(self, small_env):
        """get_agent should raise ValueError for unknown agent type."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent("nonexistent-agent", small_env)

    def test_get_agent_passes_kwargs(self, small_env):
        """get_agent should pass kwargs to agent constructor."""
        agent = get_agent("reinforce", small_env, lr=0.123, gamma=0.456)
        assert agent.gamma == 0.456

    def test_get_agent_cpt_params(self, small_env):
        """get_agent should pass CPT parameters correctly."""
        agent = get_agent(
            "cpt-reinforce",
            small_env,
            alpha=0.75,
            beta=0.85,
            lambda_=3.0,
            reference_point=-5.0,
        )
        assert agent.cpt_value.alpha == 0.75
        assert agent.cpt_value.beta == 0.85
        assert agent.cpt_value.lambda_ == 3.0
        assert agent.cpt_value.reference_point == -5.0


class TestPolicyGradientDirection:
    """Tests for policy gradient update direction."""

    def test_high_returns_increase_action_probability(self, env_factory, torch_seed):
        """Actions with high returns should have increased probability after update."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = REINFORCEAgent(env, lr=0.1, gamma=0.99)  # High LR for visible effect

        # Get initial action probabilities for state 0
        state = 0
        initial_probs = agent.policy(state).detach().clone()

        # Simulate episode where action 1 leads to high return
        agent.log_probs = []
        agent.rewards = []

        # Manually set up a "good" episode for action 1
        probs = agent.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = 1  # Force action 1
        agent.log_probs.append(dist.log_prob(torch.tensor(action)))
        agent.rewards.append(10.0)  # High positive reward

        # Compute returns and update
        returns = agent._transform_returns([10.0])
        loss = -agent.log_probs[0] * returns[0]

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # Check probability increased for action 1
        new_probs = agent.policy(state).detach()

        assert new_probs[action].item() > initial_probs[action].item(), \
            f"Probability of action {action} should increase after positive return"

    def test_negative_returns_decrease_action_probability(self, env_factory, torch_seed):
        """Actions with very negative returns should have decreased probability."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = REINFORCEAgent(env, lr=0.1, gamma=0.99)

        state = 0
        initial_probs = agent.policy(state).detach().clone()

        # Simulate episode where action 2 leads to very negative return
        agent.log_probs = []
        agent.rewards = []

        probs = agent.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = 2
        agent.log_probs.append(dist.log_prob(torch.tensor(action)))
        agent.rewards.append(-100.0)  # Very negative reward

        returns = agent._transform_returns([-100.0])
        loss = -agent.log_probs[0] * returns[0]

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        new_probs = agent.policy(state).detach()

        assert new_probs[action].item() < initial_probs[action].item(), \
            f"Probability of action {action} should decrease after negative return"
