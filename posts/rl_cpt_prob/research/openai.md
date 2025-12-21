Short answer:
you can do an analogue of “prospect theory probability weighting” in RL, but it cannot be done with a simple per-step reward transform like you do for the value function. You have to touch either:
	1.	the objective over the whole return distribution (CPT / distortion risk measures), or
	2.	the learning rule (e.g. TD error or trajectory weights),
	3.	or use a distributional critic and apply the weighting on its CDF.

Let me break it down in a way that’s directly usable for experiments.

⸻

1. Why value-function shaping is easy but probability weighting is not

Prospect Theory has two pieces:
	•	Value / utility function v(x): concave for gains, convex for losses, kink at 0.
→ In RL you can just do reward shaping: r'_t = v(r_t). Then you’re maximizing \mathbb{E}[\sum_t \gamma^t v(r_t)]. This is local and Markovian.
	•	Probability weighting w(p): nonlinear mapping of probabilities (Prelec, TK 92, etc.), typically inverse-S: overweights small p, underweights large p.  ￼

The problem: w acts on probabilities of events, i.e. on the distribution of the return, not on individual rewards. For an episode return G, the CPT value is something like (simplified)  ￼

C_{u,w}(G)
=\int_0^\infty w^+\big(P(u^+(G) > z)\big)\,dz
\;-\;\int_0^\infty w^-\big(P(u^-(G) > z)\big)\,dz

You cannot write that as “apply some function f to each immediate reward and then take an expectation”. The weights for an outcome depend on its rank in the entire distribution, not just on the local transition.

So there is no Markovian per-step reward transform that exactly reproduces CPT probability weighting.

But: there are principled ways to put w(\cdot) into RL.

⸻

2. Trajectory-level CPT objective (closest conceptual analogue to “reward shaping”)

This is exactly what Cumulative Prospect Theory Meets Reinforcement Learning does.  ￼

Idea:
	1.	Define the episodic return G(\tau) for trajectory \tau.
	2.	Define your performance as the CPT functional of the return distribution:
J_{\text{CPT}}(\pi) = C_{u,w}\big(G_\pi\big).
	3.	To estimate this from data, sample N episodes from your current policy:
	•	Collect returns G_1,\dots,G_N.
	•	Sort them (separately for gains/losses) and compute a quantile estimator for CPT:
\hat C_N = \sum_i u(G_{(i)}) \,\Delta w_i,
where \Delta w_i = w\!\left(\tfrac{N+1-i}{N}\right) - w\!\left(\tfrac{N-i}{N}\right) for gains (and a reversed version for losses).  ￼
	4.	Use this \hat C_N inside a policy search algorithm:
	•	Either SPSA (as in the paper), or
	•	REINFORCE-style gradient with trajectory weights.

For REINFORCE-style:
	•	Normally:
\nabla_\theta J(\pi_\theta) \approx \frac{1}{N}\sum_{k=1}^N \bigg(\sum_t \nabla_\theta \log\pi_\theta(a_t^{(k)}|s_t^{(k)}) \bigg) \, G_k.
	•	For CPT, intuitively you want to replace the simple G_k term by something like a rank-dependent weight:
\nabla_\theta J_{\text{CPT}}(\pi_\theta)
\approx \sum_{k=1}^N w_k^{\text{CPT}}
\bigg(\sum_t \nabla_\theta \log\pi_\theta(a_t^{(k)}|s_t^{(k)})\bigg),
where w_k^{\text{CPT}} depends on the rank of G_k and the corresponding \Delta w_i.

Rough recipe you can actually implement:
	1.	Sample batch of trajectories.
	2.	Compute returns G_k, sort them.
	3.	For each sorted index i, compute CPT decision weight increment \Delta w_i.
	4.	Map back from sorted index to the original trajectory index k and set a per-trajectory scalar weight \lambda_k := \Delta w_i.
	5.	Use:
g \leftarrow \sum_k \lambda_k \sum_t \nabla_\theta \log\pi_\theta(a_t^{(k)}|s_t^{(k)}).

This is the direct analogue of “we don’t change the environment, but we change how trajectories influence the gradient” in line with CPT’s probability weighting.

Caveat:
	•	This objective is time-inconsistent for typical CPT weightings, so there is no clean Bellman equation; you’re in pure policy search land (which is what Prashanth & Szepesvári do).  ￼

⸻

3. Distributional RL: put w on the critic’s CDF

This is, in practice, the cleanest way to combine CPT’s probability part with standard deep RL.

Set up a distributional critic Z_\theta(s,a) that outputs an approximation of the full return distribution (e.g., QR-DQN, IQN, distributional actor-critic).

Then:
	1.	From Z_\theta(s,a), get a set of quantiles \{z_k\} with base quantile levels \{\tau_k\} (e.g., \tau_k = \frac{k}{K}).
	2.	Define decision weights using your CPT probability weighting function w:
	•	For gains/losses separated or in one shot depending on how strict you want to be.
	•	Simplest: treat everything as “gains” and define
\Delta w_k = w(\tau_{k}) - w(\tau_{k-1}),\ \ \tau_0 = 0.
	3.	Define a CPT-Q value:
Q_{\text{CPT}}(s,a) \approx \sum_{k=1}^K v(z_k)\, \Delta w_k .
where v is your PT value function (you already have this from your reward shaping idea).
	4.	Use Q_{\text{CPT}}(s,a) for:
	•	Action selection: \pi(a|s) ∝ softmax(Q_{\text{CPT}}(s,a)), or greedy.
	•	Actor update: advantage A(s,a) = Q_{\text{CPT}}(s,a) - \mathbb{E}_{a'}[Q_{\text{CPT}}(s,a')].

The critic itself is still trained with standard distributional Bellman updates (C51/QR/IQN loss), which is nice: all the PT machinery lives in the head you bolt on top.

The probability distortion is then operating exactly where it should: on the CDF of the return. Mathematically this is in the family of distortion risk measures / Choquet integrals, which generalize CPT’s weighting idea and are now well-studied in risk-sensitive RL.  ￼

⸻

4. TD-error nonlinearity: “local” implementation that implicitly distorts probabilities

There’s also a clever trick that is closer in spirit to “local shaping”: apply a nonlinear utility to the TD-error. This is the path taken in Risk-Sensitive Reinforcement Learning by Shen et al.  ￼

They show:
	•	If you update with:
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad
V(s_t) \leftarrow V(s_t) + \alpha\,u(\delta_t),
for some utility u, then you effectively:
	•	transform rewards and also
	•	induce a nonlinear transformation of transition probabilities in the underlying MDP, giving rise to risk-sensitive behavior that can match prospect-theoretic curves (different risk preferences for gains/losses, nonlinear probability weighting, etc.).  ￼

So a practical recipe:
	1.	Design a piecewise u mimicking PT:
	•	steeper for negative TD errors (loss aversion),
	•	S-shaped to induce PT-like probability distortion.
	2.	Plug u(\delta_t) into your Q-learning / actor–critic updates.

This doesn’t give you an exact CPT probability weighting on the return distribution, but it locally implements similar biases and is easy to bolt onto an existing agent.

⸻

5. Dynamic distortion risk measures: “Bellman-friendly” probability weighting

CPT itself is dynamically inconsistent; but there is a related class of objectives where probability distortion is done in a time-consistent way: distortion risk measures (Choquet integrals). These can be written recursively:

Q^\pi_h(s,a) = r_h(s,a) + \rho_h\big(V^\pi_{h+1}(s')\big),

where \rho_h is a one-step dynamic risk measure applied to the distribution of the next-state value.  ￼

If you choose \rho_h as a distortion risk measure (Choquet integral with distortion g on the CDF), you get a Bellman equation that is “probability-weighted” at every step. In continuous-time RL, similar ideas appear as Choquet regularization.  ￼

Practically, this again pushes you toward distributional RL: you approximate the next-value distribution and apply \rho_h (i.e., your w) to that distribution inside Bellman backups.

⸻

6. “Can I just rewrite the reward with w somehow?”

If by “similar to reward shaping” you mean something like:

“For each transition, compute a new reward r’ that somehow encodes probability weighting.”

Then strictly speaking:
	•	No: there is no per-transition function f(s,a,s',r) that, when plugged into a standard expected-return objective, exactly reproduces CPT probability weighting on the return distribution.

What is analogous to reward shaping for probabilities is:
	•	Reweighting trajectories or TD updates according to CPT decision weights (Section 2), or
	•	Distorting the critic’s CDF before acting (Section 3), or
	•	Nonlinearly transforming TD errors (Section 4).

All three are “implementation sites” for w(\cdot).

⸻

7. What I’d do for your experiments

Given you’re already doing reward shaping with v(\cdot), I’d try in roughly this order:
	1.	Evaluation-only CPT (cheap sanity check)
	•	Keep training risk-neutral.
	•	After each training checkpoint, estimate J_{\text{CPT}}(\pi) on a batch of episodes using the quantile estimator (Algorithm 1 in Prashanth & Szepesvári).  ￼
	•	This immediately tells you whether your environments actually separate EV vs CPT.
	2.	Trajectory-weighted REINFORCE / actor-critic
	•	On-policy: use CPT decision weights as per-trajectory scalars in the policy-gradient estimate.
	•	This is the closest to “probability-shaped learning” without changing the environment.
	3.	Distributional actor–critic with CPT head
	•	Implement QR / IQN critic.
	•	Use Q_{\text{CPT}} for action selection and policy updates, as in Section 3.
	•	This is the most principled way to directly encode both value and probability parts of Prospect Theory.
	4.	TD-error utility as a simpler baseline
	•	Wrap your TD error in an S-shaped u(\cdot) and see if you already capture some PT-like risk patterns.

If you tell me what algorithm you’re currently using (vanilla policy gradient, PPO, A2C, Q-learning, distributional, etc.), I can sketch very concrete pseudo-code for one of these approaches tailored to your setup (e.g., “PPO + CPT trajectory weights” or “IQN critic + CPT policy head”).

Yes—there are ways to do a “prospect-theory style” probability weighting with PPO-type algorithms, but (unlike value-function shaping) you have to plug it into the objective / critic / weighting of samples, not into a per-step reward transform.

Let me give you three concrete patterns that are PPO-compatible, from “closest to vanilla PPO” to “more architectural change”.

⸻

1. CPT-weighted PPO: reweight trajectories in the PPO loss

Think of PPO’s policy gradient as:

\nabla J(\theta) \approx
\mathbb{E}\Big[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, A_t \Big].

PPO implements this with the clipped surrogate:

L_{\text{PPO}}(\theta)
= \mathbb{E}_t\left[
\min\!\Big(
r_t(\theta)\, \hat A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t
\Big)
\right],

where r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} and \hat A_t is usually GAE.

To inject prospect-theory probability weighting, you can:
	1.	Work with episodic returns G_k (one per trajectory).
	2.	Sort the returns G_{(1)} \le \dots \le G_{(N)}.
	3.	For each sorted index i, define a CPT decision weight increment using your weighting function w (e.g. Prelec):
For gains:
\Delta w_i = w\!\left(\frac{N+1-i}{N}\right) - w\!\left(\frac{N-i}{N}\right), \quad i=1,\dots,N
and a mirrored definition for losses, like in Prashanth & Szepesvári’s CPT-RL work.
	4.	Map each trajectory back to its original index k, and define a per-trajectory scalar weight \lambda_k := \Delta w_{\text{rank}(k)}.
	5.	Use \lambda_k as a sample weight inside PPO:
	•	Every timestep t belonging to trajectory k gets weight \lambda_k.
	•	Your surrogate loss becomes
L_{\text{CPT-PPO}}(\theta) =
\mathbb{E}_t\big[
\lambda_{\tau(t)} \,
\min\big(
r_t(\theta)\,\hat A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t
\big)
\big],
where \tau(t) is the trajectory index for timestep t.
In code (PyTorch-ish, single batch):

# returns: shape [n_traj]
returns = traj_returns.clone().detach()
sort_idx = torch.argsort(returns)           # ascending
ranks = torch.empty_like(sort_idx)
ranks[sort_idx] = torch.arange(len(returns))

# normalized probs p_i = (i+1)/N, i from 0..N-1
N = len(returns)
p = (torch.arange(N, dtype=torch.float32) + 1) / N
# w(p): your prospect weighting function
wp = w(p)                                   # shape [N]
wp_prev = torch.cat([torch.zeros(1), wp[:-1]])
delta_w = wp - wp_prev                      # decision weights

lambda_traj = delta_w[ranks]                # map back to original order

# broadcast per-trajectory λ to timesteps (batch flattened)
lambda_t = lambda_traj[traj_id_per_timestep]  # shape [T_total]

# standard PPO bits...
ratio = (logpi_new - logpi_old).exp()
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1-eps, 1+eps) * advantages

loss_pg = - (lambda_t * torch.min(unclipped, clipped)).mean()

Implementation tips:
	•	Normalize \lambda_k so that \frac{1}{N}\sum_k \lambda_k \approx 1 to keep learning rates comparable (e.g. divide by mean |λ|).
	•	If you use GAE, don’t re-normalize the advantages after multiplying by λ, or you’ll partially undo the probability weighting.

Interpretation:
instead of every trajectory contributing equally (risk-neutral), high-return or low-return trajectories get nonlinear, rank-dependent weight in the gradient—exactly the spirit of CPT’s probability weighting over the return distribution.

This keeps:
	•	on-policy PPO data collection,
	•	clipped updates,
	•	your existing GAE machinery,

and just changes how trajectories are aggregated.

⸻

2. Distributional PPO + CPT head (distortion risk measures ≈ CPT probability weighting)

Distributional RL is a very natural place to put probability weighting, because you directly manipulate the return distribution instead of just its mean. Distortion risk measures, which include cumulative prospect theory as a special case, are implemented precisely by “warping” the CDF.

Recipe for a “CPT-PPO with distributional critic”:
	1.	Replace your scalar critic V_\phi(s) with a distributional critic:
	•	QR-style: outputs K quantiles \{z_k(s)\} at fixed fractions \tau_k = \frac{k}{K+1}.
	•	Or IQN: samples quantile fractions \tau\sim U(0,1) and outputs z(s,\tau).
	2.	Train this critic with a standard distributional TD loss (C51/QR/IQN), independent of CPT.
	3.	To feed the actor, build a CPT-style Q or V by integrating over quantiles with distorted weights:
	•	Compute decision weights:
\Delta w_k = w(\tau_k) - w(\tau_{k-1}), \quad \tau_0=0
	•	Then define a “CPT value” at state s (simplest, gains only):
V_{\text{CPT}}(s) \approx \sum_{k=1}^K v\big(z_k(s)\big) \, \Delta w_k.
	•	For an actor–critic, you’d usually want action-values:
either estimate a distributional Z(s,a) or sample one-step ahead. Then:
Q_{\text{CPT}}(s,a) \approx \sum_{k=1}^K v\big(z_k(s,a)\big)\,\Delta w_k.
	4.	Use V_{\text{CPT}} / Q_{\text{CPT}} in PPO:
	•	Define advantages:
A_t^{\text{CPT}} = Q_{\text{CPT}}(s_t,a_t) - \mathbb{E}_{a'}[Q_{\text{CPT}}(s_t,a')]
or state-value version if you prefer.
	•	Plug A_t^{\text{CPT}} into the usual clipped PPO objective.

This is exactly what a bunch of risk-sensitive distributional RL papers do for CVaR, Wang distortion, etc.—all of which are special cases of distortion risk measures that cover cumulative prospect theory’s probability weighting.

Conceptually:
	•	The critic learns a risk-neutral return distribution.
	•	The actor uses a prospect-distorted summary of this distribution Q_{\text{CPT}} to decide how to act.

You keep PPO’s structure, just swapping in a fancier critic.

⸻

3. Prospect-style nonlinearity on the TD error in PPO’s critic

Less “exactly CPT”, more “PT-flavoured risk”:
you can implement a nonlinear utility on the TD error in the critic, which has been analysed as inducing risk-sensitive / probability-distorted behavior.

For PPO, the critic loss is usually:

L_V(\phi) = \mathbb{E}_t\big[ (V_\phi(s_t) - \hat G_t)^2 \big].

Instead, define a PT-style utility u(\delta) (steeper for negative errors, S-shaped, etc.), and update:
	•	TD error: \delta_t = \hat G_t - V_\phi(s_t).
	•	Loss: L_V(\phi) = \mathbb{E}_t[ u(\delta_t)^2 ] or even a gradient step proportional to u(\delta_t).

This changes:
	•	how the critic learns from under/over-estimations,
	•	implicitly distorting the effective transition probabilities and reward sensitivities in a way that can approximate PT-like behavior, without rewriting the environment or the PPO policy loss.

This is simpler to bolt on, but it’s a more indirect, “local” approximation to PT probability weighting.

⸻

Which should you try first?

Given you already have PPO infrastructure:
	1.	Easiest drop-in (no new networks):
CPT-weighted PPO via trajectory weights (Option 1)
	•	One extra post-processing over returns per batch.
	•	No change in architecture, just sample weights in the loss.
	•	Good for proof-of-concept “EV vs CPT” comparisons in your toy envs.
	2.	More principled & expressive, but more work:
Distributional critic + CPT head (Option 2)
	•	If you’re comfortable with QR/IQN, this is the cleanest way to literally apply PT’s w(p) on the return CDF.
	3.	Quick hack to make PPO more risk-aware:
Nonlinear TD utility (Option 3)
	•	Minimal change, but more heuristic if your goal is explicit CPT.

If you tell me which PPO variant you’re running right now (vanilla, clipped w/ GAE, distributional, etc.) and whether you’re on-policy episodic or continuing control, I can write you concrete pseudo-code for one of these (e.g., “CPT-weighted PPO for finite-horizon episodic control with GAE”) that you can drop into your current codebase.