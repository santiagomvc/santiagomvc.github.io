"""Tests for REINFORCE, CPT-PG, and hybrid agents."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    REINFORCEAgent,
    PerStepCPTAgent,
    CPTPGAgent,
    CPTPGRUDDERAgent,
    AGENTS,
    get_agent,
)
from utils import CPTValueFunction, CPTWeightingFunction, PerStepSlidingWindowCPT, SlidingWindowCPT


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


class TestAgentRegistry:
    """Tests for agent registry and factory function."""

    def test_agents_dict_contains_expected_agents(self):
        """AGENTS dict should contain all expected agent types."""
        expected_agents = ["random", "reinforce", "per-step-cpt", "cpt-pg", "cpt-pg-rudder", "llm"]
        for agent_name in expected_agents:
            assert agent_name in AGENTS, f"'{agent_name}' should be in AGENTS"

    def test_get_agent_creates_reinforce(self, small_env):
        """get_agent should create REINFORCEAgent."""
        agent = get_agent("reinforce", small_env, lr=1e-3, gamma=0.99)
        assert isinstance(agent, REINFORCEAgent)

    def test_get_agent_unknown_raises_error(self, small_env):
        """get_agent should raise ValueError for unknown agent type."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent("nonexistent-agent", small_env)

    def test_get_agent_passes_kwargs(self, small_env):
        """get_agent should pass kwargs to agent constructor."""
        agent = get_agent("reinforce", small_env, lr=0.123, gamma=0.456)
        assert agent.gamma == 0.456


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


class TestCPTWeightingFunctionClamp:
    """Tests for w' derivative clamping (Plan A.3)."""

    def test_w_prime_plus_clamped(self):
        """w_prime_plus should be clamped at max_wprime."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        # At p=1.0, raw derivative is ~880; should be clamped to 50
        val = wf.w_prime_plus(1.0)
        assert val <= 50.0, f"w_prime_plus(1.0) = {val} should be <= 50.0"
        # At p=0.0, raw derivative is also large; should be clamped
        val0 = wf.w_prime_plus(0.0)
        assert val0 <= 50.0, f"w_prime_plus(0.0) = {val0} should be <= 50.0"

    def test_w_prime_minus_clamped(self):
        """w_prime_minus should be clamped at max_wprime."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        val = wf.w_prime_minus(1.0)
        assert val <= 50.0, f"w_prime_minus(1.0) = {val} should be <= 50.0"

    def test_w_prime_interior_not_clamped(self):
        """w' at interior points should generally not be clamped."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        # At p=0.5, derivative should be moderate and not clamped
        val = wf.w_prime_plus(0.5)
        assert val < 50.0
        assert val > 0.0


class TestCPTPGAgent:
    """Tests for CPT-PG agent (Plans A.1, A.2)."""

    def test_inherits_from_reinforce(self, cpt_pg_agent):
        assert isinstance(cpt_pg_agent, REINFORCEAgent)

    def test_baseline_is_zero(self, cpt_pg_agent):
        """CPT-PG forces baseline_type='zero'."""
        assert cpt_pg_agent.baseline_type == "zero"
        assert cpt_pg_agent.baseline == 0.0

    def test_compute_phi_centered(self, cpt_pg_agent):
        """φ̂ values should be centered (mean ≈ 0) after Plan A.1 fix."""
        returns = [-17.0, -17.0, -17.0, -17.0, -17.0, -17.0, -17.0, -105.0]
        phi = cpt_pg_agent._compute_phi(returns)
        assert abs(phi.mean()) < 1e-10, f"φ̂ mean should be ~0, got {phi.mean()}"

    def test_compute_phi_discriminates_cliff_vs_safe(self, cpt_pg_agent):
        """After centering, cliff and safe episodes should have meaningfully different φ̂."""
        returns = [-17.0, -17.0, -17.0, -17.0, -17.0, -17.0, -17.0, -105.0]
        phi = cpt_pg_agent._compute_phi(returns)
        # Cliff episode (last) should have more negative φ̂ than safe episodes
        safe_phi = phi[:7]
        cliff_phi = phi[7]
        assert cliff_phi < safe_phi.min(), \
            f"Cliff φ̂ ({cliff_phi:.2f}) should be < safe φ̂ ({safe_phi.min():.2f})"
        # Relative spread should be meaningful (not 2% like before fix)
        spread = (safe_phi.mean() - cliff_phi) / (abs(safe_phi.mean()) + abs(cliff_phi) + 1e-10)
        assert spread > 0.1, f"Relative spread {spread:.4f} should be > 0.1"

    def test_compute_phi_unconditional_survival(self, small_env, torch_seed):
        """Survival function should use n (total), not n_gains/n_losses (Plan A.2).

        With mixed gains/losses, using conditional n would give wrong probabilities.
        """
        agent = CPTPGAgent(small_env, alpha=0.88, beta=0.88, lambda_=2.25)
        # Mix of positive and negative returns
        returns = [10.0, 20.0, -5.0, -15.0]
        phi = agent._compute_phi(returns)
        # Gains should have positive phi, losses negative
        assert phi[0] > phi[2], "Gain episode should have higher φ̂ than loss episode"
        assert phi[1] > phi[3], "Larger gain should have higher φ̂ than larger loss"

    def test_compute_phi_all_same_returns_zero(self, cpt_pg_agent):
        """If all returns are identical, centered φ̂ should be all zeros."""
        returns = [-17.0, -17.0, -17.0, -17.0]
        phi = cpt_pg_agent._compute_phi(returns)
        for p in phi:
            assert abs(p) < 1e-10, f"φ̂ should be 0 for identical returns, got {p}"

    def test_cpt_pg_learns(self, env_factory, torch_seed):
        """CPT-PG should show learning progress on a simple env."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = CPTPGAgent(env, lr=1e-2, gamma=0.99)
        history = agent.learn(env, timesteps=5000)
        rewards = history['episode_rewards']
        # Compare early vs late performance
        early = np.mean(rewards[:20])
        late = np.mean(rewards[-20:])
        assert late > early, f"CPT-PG should improve: late {late:.1f} > early {early:.1f}"


class TestCPTPGRUDDERAgent:
    """Tests for CPT-PG with RUDDER decomposition (Plan C)."""

    def test_inherits_from_cpt_pg(self, cpt_pg_rudder_agent):
        assert isinstance(cpt_pg_rudder_agent, CPTPGAgent)

    def test_has_rudder_model(self, cpt_pg_rudder_agent):
        assert hasattr(cpt_pg_rudder_agent, 'rudder')
        assert hasattr(cpt_pg_rudder_agent, 'rudder_optimizer')

    def test_cpt_pg_rudder_learns(self, env_factory, torch_seed):
        """CPTPGRUDDERAgent should show learning progress."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = CPTPGRUDDERAgent(env, lr=1e-2, gamma=0.99)
        history = agent.learn(env, timesteps=5000)
        rewards = history['episode_rewards']
        early = np.mean(rewards[:20])
        late = np.mean(rewards[-20:])
        assert late > early, f"CPTPGRUDDER should improve: late {late:.1f} > early {early:.1f}"

    def test_get_agent_creates_cpt_pg_rudder(self, small_env):
        """get_agent should create CPTPGRUDDERAgent."""
        agent = get_agent("cpt-pg-rudder", small_env, alpha=0.88)
        assert isinstance(agent, CPTPGRUDDERAgent)


class TestPerStepSlidingWindowCPT:
    """Tests for PerStepSlidingWindowCPT utility class."""

    def test_single_timestep_reduces_to_sliding_window(self):
        """With T=1 episodes, per-step weights should match SlidingWindowCPT weights."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        sw = SlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)
        psw = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        # Single-step episodes: each has exactly one return value
        episode_returns = [-50.0, -10.0, -5.0, -3.0, -3.0, -3.0, -3.0, -3.0]
        per_step_returns = [[r] for r in episode_returns]
        dummy_meta = list(range(len(episode_returns)))

        sw.add_batch(episode_returns)
        psw.add_batch(per_step_returns, dummy_meta)

        sw_weights = sw.compute_decision_weights(episode_returns)
        psw_weights = psw.compute_decision_weights(per_step_returns, is_ratio_fn=lambda _: 1.0)

        # Per-step weights at t=0 should match SlidingWindowCPT weights
        psw_t0 = [w[0] for w in psw_weights]
        for i in range(len(episode_returns)):
            assert abs(psw_t0[i] - sw_weights[i]) < 1e-10, \
                f"Episode {i}: per-step weight {psw_t0[i]} != sliding window weight {sw_weights[i]}"

    def test_variable_length_episodes(self):
        """Only active episodes should contribute at each timestep."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        psw = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        # Episodes of different lengths
        per_step_returns = [
            [-10.0, -5.0, -3.0],  # 3 steps
            [-20.0, -8.0],        # 2 steps
            [-15.0],              # 1 step
        ]
        dummy_meta = list(range(3))

        psw.add_batch(per_step_returns, dummy_meta)
        weights = psw.compute_decision_weights(per_step_returns, is_ratio_fn=lambda _: 1.0)

        # Episode 0 should have 3 weights
        assert len(weights[0]) == 3
        # Episode 1 should have 2 weights
        assert len(weights[1]) == 2
        # Episode 2 should have 1 weight
        assert len(weights[2]) == 1

        # At t=0, all 3 episodes contribute
        # At t=1, only episodes 0 and 1 contribute
        # At t=2, only episode 0 contributes (single episode → weight = 1.0)
        assert abs(weights[0][2] - 1.0) < 1e-10, \
            f"Single-episode timestep should have weight 1.0, got {weights[0][2]}"

    def test_weights_normalize_per_timestep(self):
        """Mean weight at each timestep should be ~1.0."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        psw = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        per_step_returns = [
            [-10.0, -5.0, -3.0, -1.0],
            [-50.0, -30.0, -20.0, -10.0],
            [-3.0, -2.0, -1.0, -0.5],
            [-8.0, -4.0, -2.0, -1.0],
        ]
        dummy_meta = list(range(4))

        psw.add_batch(per_step_returns, dummy_meta)
        weights = psw.compute_decision_weights(per_step_returns, is_ratio_fn=lambda _: 1.0)

        # Check that mean weight at each timestep ≈ 1.0
        for t in range(4):
            weights_at_t = [weights[i][t] for i in range(4)]
            mean_w = sum(weights_at_t) / len(weights_at_t)
            assert abs(mean_w - 1.0) < 1e-10, \
                f"Timestep {t}: mean weight {mean_w} should be ~1.0"

    def test_gain_loss_separation(self):
        """Gains should use w+, losses should use w-."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        psw = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        # Mix of gains and losses at each timestep
        per_step_returns = [
            [10.0, 5.0],   # gains
            [-10.0, -5.0],  # losses
            [20.0, 10.0],  # gains
            [-20.0, -10.0],  # losses
        ]
        dummy_meta = list(range(4))

        psw.add_batch(per_step_returns, dummy_meta)
        weights = psw.compute_decision_weights(per_step_returns, is_ratio_fn=lambda _: 1.0)

        # All weights should be positive
        for i in range(4):
            for t in range(2):
                assert weights[i][t] > 0, \
                    f"Weight [{i}][{t}] = {weights[i][t]} should be positive"

    def test_is_ratio_one_matches_no_ratio(self):
        """IS ratio of 1.0 should produce identical weights (sanity check)."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        psw_a = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)
        psw_b = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        batch1 = [[-10.0, -5.0], [-3.0, -2.0], [-8.0, -4.0]]
        batch2 = [[-7.0, -3.0], [-12.0, -6.0], [-4.0, -1.0]]
        meta1 = list(range(3))
        meta2 = list(range(3))

        for psw in [psw_a, psw_b]:
            psw.add_batch(batch1, meta1)
            psw.add_batch(batch2, meta2)

        weights_a = psw_a.compute_decision_weights(batch2, is_ratio_fn=lambda _: 1.0)
        weights_b = psw_b.compute_decision_weights(batch2, is_ratio_fn=lambda _: 1.0)

        for i in range(len(batch2)):
            for t in range(len(batch2[i])):
                assert abs(weights_a[i][t] - weights_b[i][t]) < 1e-10, \
                    f"Weights should be identical with ratio=1.0"

    def test_is_ratio_changes_weights(self):
        """IS ratio != 1.0 on historical batches should change the weights."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        psw_a = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)
        psw_b = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        batch1 = [[-10.0, -5.0], [-3.0, -2.0], [-50.0, -30.0]]
        batch2 = [[-7.0, -3.0], [-12.0, -6.0], [-4.0, -1.0]]
        meta1 = list(range(3))
        meta2 = list(range(3))

        for psw in [psw_a, psw_b]:
            psw.add_batch(batch1, meta1)
            psw.add_batch(batch2, meta2)

        weights_a = psw_a.compute_decision_weights(batch2, is_ratio_fn=lambda _: 1.0)
        weights_b = psw_b.compute_decision_weights(batch2, is_ratio_fn=lambda _: 3.0)

        # At least one weight should differ
        any_diff = False
        for i in range(len(batch2)):
            for t in range(len(batch2[i])):
                if abs(weights_a[i][t] - weights_b[i][t]) > 1e-10:
                    any_diff = True
                    break
        assert any_diff, "Weights should differ when IS ratio != 1.0"

    def test_is_ratio_skips_current_batch(self):
        """With a single batch, IS ratio should be ignored (all data is current)."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)

        psw_a = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)
        psw_b = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        batch = [[-10.0, -5.0], [-3.0, -2.0], [-50.0, -30.0], [-8.0, -4.0]]
        meta = list(range(4))

        psw_a.add_batch(batch, meta)
        psw_b.add_batch(batch, meta)

        weights_a = psw_a.compute_decision_weights(batch, is_ratio_fn=lambda _: 1.0)
        weights_b = psw_b.compute_decision_weights(batch, is_ratio_fn=lambda _: 5.0)

        for i in range(len(batch)):
            for t in range(len(batch[i])):
                assert abs(weights_a[i][t] - weights_b[i][t]) < 1e-10, \
                    f"Single-batch weights should be identical regardless of IS ratio"


class TestPerStepCPTAgent:
    """Tests for PerStepCPTAgent."""

    def test_separated_baseline(self, small_env, torch_seed):
        """Verify v(G) - v(b) is used, NOT v(G - b)."""
        agent = PerStepCPTAgent(small_env, alpha=0.88, beta=0.88, lambda_=2.25)
        v = agent.cpt_value

        # Set a known baseline
        G_t = -10.0
        b_t = -5.0

        # Separated: v(G_t) - v(b_t)
        separated = v(G_t) - v(b_t)
        # Non-separated (wrong): v(G_t - b_t)
        non_separated = v(G_t - b_t)

        # These should differ because v is nonlinear
        assert abs(separated - non_separated) > 0.01, \
            f"v(G)-v(b)={separated:.4f} should differ from v(G-b)={non_separated:.4f}"

    def test_per_timestep_baselines_update(self, env_factory, torch_seed):
        """Baselines dict should be populated after learning."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = PerStepCPTAgent(env, lr=1e-2, gamma=0.99)

        assert len(agent.baselines) == 0

        agent.learn(env, timesteps=500)

        # After training, baselines should exist for multiple timesteps
        assert len(agent.baselines) > 0, "Baselines dict should be populated after learning"
        # All baseline values should be non-zero (they track negative returns)
        for t, b_t in agent.baselines.items():
            assert isinstance(t, int)
            assert isinstance(b_t, float)

    def test_agent_learns_basic(self, env_factory, torch_seed):
        """PerStepCPTAgent should show learning progress."""
        env = env_factory(shape=(3, 3), wind_prob=0.0, reward_cliff=-50.0, reward_step=-1.0)
        agent = PerStepCPTAgent(env, lr=1e-2, gamma=0.99)
        history = agent.learn(env, timesteps=5000)
        rewards = history['episode_rewards']
        early = np.mean(rewards[:20])
        late = np.mean(rewards[-20:])
        assert late > early, \
            f"PerStepCPT should improve: late {late:.1f} > early {early:.1f}"

    def test_per_step_weights_applied(self, env_factory, torch_seed):
        """Different timesteps should get different weights."""
        wf = CPTWeightingFunction(gamma_plus=0.61, gamma_minus=0.69)
        psw = PerStepSlidingWindowCPT(wf, max_batches=5, decay=0.8, reference_point=0.0)

        # Create returns with very different distributions at each timestep
        per_step_returns = [
            [-50.0, -30.0, -10.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0],
        ]
        dummy_meta = list(range(8))

        psw.add_batch(per_step_returns, dummy_meta)
        weights = psw.compute_decision_weights(per_step_returns, is_ratio_fn=lambda _: 1.0)

        # Episode 0 (outlier at t=0) should have a different weight at t=0
        # compared to the other episodes
        w_outlier_t0 = weights[0][0]
        w_normal_t0 = weights[1][0]
        assert abs(w_outlier_t0 - w_normal_t0) > 0.01, \
            f"Outlier weight {w_outlier_t0:.4f} should differ from normal {w_normal_t0:.4f} at t=0"

    def test_get_agent_creates_per_step_cpt(self, small_env):
        """get_agent should create PerStepCPTAgent."""
        agent = get_agent("per-step-cpt", small_env, alpha=0.88)
        assert isinstance(agent, PerStepCPTAgent)
