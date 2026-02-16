---
title: "AI and the irrationality of human behavior under risk"
subtitle: "Comparing AI and Human decision making under risk"
date: 2026-02-14
categories: [reinforcement-learning, behavioral-economics, prospect-theory]
toc: true
---

Current AI systems are undoublty able to make some decisions and perform some actions humans usually do. However how those decisions are made is likely pretty different. Language Models are trained with incredible large amounts of textual data, creating base models that are then aligned to perform as useful assistants with multiple methods like SFT, RLHF, RLVR. The ouptput is an assistant that is indeed useful and rational in most cases. Human decision making on the other side, while still largely a black box, has continuos demonstrations of irrationality and biases, with experiments replicating for different populations and decision making scenarios. 

In this post we compare the the decision making of some AI systems (Pure RL, LLM) with an approximation of the expected behavior of humans making decisions under risk. We will use a custom cliff walking environment to train and evaluate the agents. We will talk about the technical details on the next section, for this first one just try to put yourself in the position of the agent and imagine how you would make the decision.

---

## The Setup

You are an agent in a grid world. Your goal is to reach the target position on the other side. The bottom edge of the grid is a cliff, fall off and the episode ends. At each step, a random wind may push you one row downward toward the cliff. Higher rows are safer (farther from the edge) but require more steps, which can penalized. The central question: **which path do you choose?**

Each experiment configures rewards, wind strength, and grid size to create a different risk profile, isolating a specific aspect of how humans deviate from rational decision-making.

## Experiment 1: The Certainty Effect

<!-- Config: exp1_hp_gains | 5x5, step=0, goal=100, cliff=5, wind=0.05, gamma=0.90, ref=25 -->

**Environment**: 5x5 grid. Steps cost nothing. Reaching the goal: +100. Falling off the cliff: +5. Wind pushes you down with 5% probability each step.

![Initial environment](images/env_exp1.png)

The shorter path hugs the cliff edge and reaches the goal faster, yielding a higher reward. The longer path climbs higher, taking more steps through the wind but staying far from the cliff. **Which path would you take?**

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp1_hp_gains_seed1/eval_4ep.gif){group="exp1"}

![CPT-PG](outputs/cpt-pg_exp1_hp_gains_seed1/eval_4ep.gif){group="exp1"}

![LLM Agent](outputs/llm_exp1_hp_gains/eval.gif){group="exp1"}
:::

**Result**: REINFORCE locks onto the riskiest path (Path1 = 100%). CPT-PG shifts decisively toward safety (Path2 = 89%, Path1 = 11%). Mann-Whitney *p* = 0.020, Cohen's *d* = 16.7, Cramer's *V* = 0.88.

![Path distribution for Experiment 1](images/exp1_path_distribution.png){#fig-exp1}

With no step cost and a large goal reward, the risky path close to the cliff is the expected-value-optimal choice --- fewer steps means less discounting, and the wind risk is low enough that the higher reward more than compensates for the occasional cliff fall.

CPT-PG avoids this path through the *certainty effect*: when gains are probable, people become risk-averse. Diminishing sensitivity compresses the reward difference between paths, while probability underweighting makes the high success rate on the risky path *feel* less certain. Together, these push the agent toward the safer option, even though it has lower expected value.


## Experiment 2: The Lottery Ticket

<!-- Config: exp3_lp_gains | 7x7, step=-1, cliff=3, goal=100, wind=0.11, gamma=0.90, ref=40.75 -->

**Environment**: 7x7 grid. Each step costs -1. Reaching the goal: +100. Falling off the cliff: +3. Wind pushes you down with 11% probability each step.

![Initial environment](images/env_exp2.png)

The risky path runs close to the cliff, reaching the goal in fewer steps for a much higher net reward --- but wind makes falling a real possibility. The safe path stays higher, giving up reward for reliability. **Which path would you take?**

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp3_lp_gains_seed1/eval_4ep.gif){group="exp2"}

![CPT-PG](outputs/cpt-pg_exp3_lp_gains_seed1/eval_4ep.gif){group="exp2"}

![LLM Agent](outputs/llm_exp3_lp_gains/eval.gif){group="exp2"}
:::

**Result**: REINFORCE never takes the risky path (Path 1 = 0%, Path 2 = 100%). CPT-PG gambles on the risky path 28% of the time (Path 1 = 28%, Path 2 = 72%), accepting a higher cliff rate in pursuit of the bigger payoff. *t*-test *p* = 0.000001, Cohen's *d* = 4.04.

![Path distribution for Experiment 2](images/exp3_path_distribution.png){#fig-exp2}

With a per-step cost and strong wind, the risky path close to the cliff offers a much higher reward when it succeeds, but the high chance of falling makes the safer path the better bet in expectation. The rational agent correctly identifies this and always plays it safe.

CPT-PG shows the *lottery ticket effect*: when gains are improbable, people become risk-seeking. CPT overweights the chance of the big payoff while diminishing sensitivity compresses the cliff penalty, making the gamble feel worth it --- just as humans buy lottery tickets despite negative expected value.


## Experiment 3: The Insurance Policy

<!-- Config: exp5_loss_aversion | 4x8, step=-1, goal=-1, cliff=-30, wind=0.10, gamma=0.98, ref=-20 -->
<!-- Comparison: REINFORCE vs CPT-PG(lambda=1.0) to isolate probability overweighting -->

**Environment**: 4x8 grid. Each step costs -1. Reaching the goal: -1. Falling off the cliff: -30. Wind pushes you down with 10% probability each step.

![Initial environment](images/env_exp3.png)

Every path costs you. The shorter path minimizes step costs but the wind can push you off the cliff for a heavy penalty. The longer path adds more step costs but keeps you far from the edge. **Which path would you take?**

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp5_loss_aversion_seed1/eval_4ep.gif){group="exp3"}

![CPT-PG ($\lambda$=1.0)](outputs/cpt-pg_lambda_1.0_exp5_loss_aversion_seed1/eval_4ep.gif){group="exp3"}

![LLM Agent](outputs/llm_exp5_loss_aversion/eval.gif){group="exp3"}
:::

**Result**: REINFORCE favors the moderately risky path (Path2 = 57%). CPT-PG with $\lambda = 1.0$ shifts toward the safest path (Path3 = 58%). Mann-Whitney *p* = 0.009, Cohen's *d* = 1.07.

![Path distribution for Experiment 3](images/exp4_path_distribution.png){#fig-exp3}

In this cost-minimizing environment, the moderately risky path balances step costs against cliff risk and comes out ahead in expected value. The rational REINFORCE agent finds this tradeoff and favors it.

To isolate the *insurance effect*, we compare REINFORCE against CPT-PG with $\lambda = 1.0$ (no loss aversion), so the only active CPT mechanism is probability weighting. CPT's weighting function inflates small probabilities: the cliff risk *feels* more dangerous than it actually is, pushing the agent toward the safest path --- just as humans buy insurance to avoid rare catastrophes, even when the expected value of the insurance is negative.


## Experiment 4: Losses Loom Larger

<!-- Config: exp5_loss_aversion (same environment as Exp 3) -->
<!-- Comparison: CPT-PG(lambda=1.0) vs CPT-PG(lambda=2.25) to isolate loss aversion -->

**Environment**: Same 4x8 grid as Experiment 3. Step: -1. Goal: -1. Cliff: -30. Wind: 10%.

This is the same environment as Experiment 3. **Does making losses feel worse push the agent even further toward safety?**

::: {layout-ncol=4}
![REINFORCE](outputs/reinforce_exp5_loss_aversion_seed1/eval_4ep.gif){group="exp4"}

![CPT-PG ($\lambda$=1.0)](outputs/cpt-pg_lambda_1.0_exp5_loss_aversion_seed1/eval_4ep.gif){group="exp4"}

![CPT-PG ($\lambda$=2.25)](outputs/cpt-pg_exp5_loss_aversion_seed1/eval_4ep.gif){group="exp4"}

![LLM Agent](outputs/llm_exp5_loss_aversion/eval.gif){group="exp4"}
:::

**Result**: CPT-PG with $\lambda = 1.0$ already shifts toward safety due to probability overweighting (Path3 = 58%). Adding loss aversion with $\lambda = 2.25$ amplifies the shift dramatically (Path3 = 79%). CPT($\lambda$=1.0) vs CPT($\lambda$=2.25): *p* = 0.005, *d* = 1.18. Full CPT vs REINFORCE: *p* = 7.4 $\times$ 10^-6^, *d* = 2.38.

![Path distribution for Experiment 4](images/exp5_path_distribution.png){#fig-exp4}

Experiments 3 and 4 decompose two CPT mechanisms in the same environment. With a reference point at $-20$, success returns land above it (perceived as gains) while cliff returns fall below (perceived as losses). The rational REINFORCE agent sees no difference --- outcomes are the same regardless of framing.

This tests Kahneman and Tversky's most famous finding: **losses hurt roughly twice as much as equivalent gains feel good**. With $\lambda = 2.25$, the cliff penalty is amplified --- a loss of $12$ below the reference *feels* like a loss of $27$. The result is a dramatic further shift toward safety, on top of the probability overweighting already seen in Experiment 3.


## How It Works

### The Grid World

The environment is a resizable grid world built on Gymnasium's CliffWalking. The agent starts at the bottom-left and must reach the bottom-right (the goal). The bottom row is the cliff: falling off ends the episode with a configurable penalty. Wind is a stochastic perturbation --- at each step, with probability `wind_prob`, the agent's action is replaced with a downward push toward the cliff. Higher rows are safer but require more steps to traverse. Rewards for stepping, reaching the goal, and falling off the cliff are all independently configurable, letting us construct pure gains domains (positive goal, zero step cost), pure losses domains (negative step cost, negative cliff penalty), and mixed domains.

### REINFORCE: The Rational Agent

REINFORCE is a Monte Carlo policy gradient algorithm. The agent runs complete episodes, computes discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$, and updates its policy by ascending the gradient $\nabla_\theta J = \mathbb{E}\left[\sum_t (G_t - b) \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$. An exponential moving average baseline $b$ reduces variance. Entropy regularization encourages exploration early and is annealed down during training. The policy network is a two-layer MLP mapping one-hot state encodings to action probabilities. This is the "rational" agent: it converges to the policy that maximizes expected discounted return.

### Cumulative Prospect Theory

Cumulative Prospect Theory models four systematic deviations from rational choice:

1. **Diminishing sensitivity**: The value function is concave for gains and convex for losses ($v(x) = x^\alpha$ for $x \geq 0$, $v(x) = -\lambda|x|^\beta$ for $x < 0$), so each additional dollar matters less.
2. **Loss aversion**: Losses are amplified by $\lambda \approx 2.25$, so losing \$100 feels as painful as gaining \$225 feels good.
3. **Probability weighting**: Small probabilities are overweighted and large probabilities are underweighted via $w(p) = p^\gamma / (p^\gamma + (1-p)^\gamma)^{1/\gamma}$.
4. **Reference dependence**: Outcomes are evaluated as gains or losses relative to a reference point, not in absolute terms.

These four components predict a "fourfold pattern" of risk attitudes: risk-averse for likely gains (Experiment 1), risk-seeking for unlikely gains (Experiment 2), risk-averse for unlikely losses (Experiment 3), and increasingly risk-averse when losses are amplified by loss aversion (Experiment 4).

### CPT-PG: Making REINFORCE Human

CPT-PG ([Lepel & Barakat, 2024](https://arxiv.org/abs/2410.02605)) replaces REINFORCE's raw returns with CPT-distorted weights $\hat{\varphi}$ computed from the batch of trajectories. For each episode $i$, the algorithm computes the discounted return $R_i$, applies the CPT value function to split it into gains $u^+$ and losses $u^-$ relative to the reference point, then integrates against the probability-weighted empirical survival function to obtain $\hat{\varphi}(R_i)$. This scalar replaces $G_t - b$ in the policy gradient. The `center_phi` normalization subtracts the batch mean $\hat{\varphi}$ to ensure the CPT-induced preference ordering is preserved even in pure domains where all $\hat{\varphi}$ values would otherwise have the same sign.

All experiments use Tversky and Kahneman's original parameter estimates: $\alpha = \beta = 0.88$, $\lambda = 2.25$, $\gamma^+ = 0.61$, $\gamma^- = 0.69$.


## Conclusions

Four experiments, four CPT mechanisms, four predicted behavioral shifts --- all confirmed. The standard REINFORCE agent behaves as expected value theory predicts, always converging to the policy with the highest expected return. The CPT-PG agent, equipped with the same neural network and training procedure but distorted by human-like biases, systematically deviates in the exact directions that Cumulative Prospect Theory predicts.

These results suggest that prospect theory's biases are not merely descriptive labels but *functional specifications* that can be engineered into artificial agents. The CPT-PG algorithm provides a principled way to build agents whose risk preferences match human behavioral patterns, which could be valuable for human-AI collaboration, preference alignment, or modeling human decision-making in economic simulations.

The code for all experiments is available at [TODO: repository link].


## References

- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5(4), 297-323.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-292.
- Lepel, T., & Barakat, A. (2024). CPT-PG: Cumulative Prospect Theory in Policy Gradient. *arXiv:2410.02605*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229-256.
