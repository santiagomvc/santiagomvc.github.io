---
title: "AI and the irrationality of human behavior under risk"
subtitle: "Comparing AI and Human decision making under risk"
date: 2026-02-14
categories: [reinforcement-learning, behavioral-economics, prospect-theory]
toc: true
---

Current AI systems are undoublty able to make some decisions and perform some actions humans usually do. However how those decisions are made is likely pretty different. Language Models are trained with incredible large amounts of textual data, creating base models that are then aligned to perform as useful assistants with multiple methods like SFT, RLHF, RLVR. The ouptput is an assistant that is indeed useful and rational in most cases. Human decision making on the other side, while still largely a black box, has continuos demonstrations of irrationality and biases, with experiments replicating for different populations and decision making scenarios. 

In this post we compare the the decision making of some AI systems (Pure RL, LLM) with an approximation of the expected behavior of humans making decisions under risk. We will use a custom cliff walking environment to train and evaluate the agents. We will talk about the technical details on the next section, for this first one just try to put yourself in the position of the agent and imagine how you would make the decision.

<!-- Current AI systems are undoubtedly able to perform actions humans do. But *how* they make decisions is fundamentally different. Standard reinforcement learning agents maximize expected value --- they are perfectly rational in the economic sense. Humans, on the other hand, systematically deviate from rationality when facing risky choices. We overvalue certainty, overweight small probabilities of disaster, and feel losses roughly twice as much as equivalent gains. These biases are captured by Cumulative Prospect Theory (CPT), the Nobel Prize-winning framework by Kahneman and Tversky.

This project explores what happens when you bake human cognitive biases directly into a reinforcement learning agent. Using a custom cliff-walking grid world, I train three types of agents --- a standard REINFORCE agent (the "rational" optimizer), a CPT-PG agent (REINFORCE with human-like distortions), and a GPT-5-mini LLM agent (to see how language models compare) --- and show that CPT-PG reliably produces the behavioral shifts predicted by decades of behavioral economics research. -->

---

## Experiments Setup

You are an agent in a grid world with a cliff and a desired position. You can move up, down, left, right. Your goal is to reach the final position with the minimum number of steps. In the process you may fall off the cliff, either due to a bad decision or due to a random wind that pushes you down. Falling off the cliff ends the episode. In negative only scenarios, taking steps costs you. In positive scenarios, taking steps reduces the final reward. The main question is: what path would you choose in each given scenario?

### Experiment 1

<starting gif frame>
<main rewards and params table>


::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp1_hp_gains_seed1/eval.gif){group="exp1"}

![CPT-PG](outputs/cpt-pg_exp1_hp_gains_seed1/eval.gif){group="exp1"}

![GPT-5-mini](outputs/llm_exp1_hp_gains_seed1/eval.gif){group="exp1"}
:::

CPT's diminishing sensitivity (the S-shaped value function) means the difference between +400 and +500 feels smaller than between +0 and +100. Combined with probability underweighting of the high success probability, CPT makes the agent prefer the "sure thing" --- the safer path with a slightly lower expected payoff --- just as a human would choose a guaranteed \$400 over an 80% chance at \$500.

**Result**: REINFORCE consistently takes the risky path (mean row 2.0). CPT-PG shifts toward safer paths (mean row 1.67). *p* = 9.75 x 10^-25^, *d* = 0.88.


### Experiment 2

<starting gif frame>
<main rewards and params table>

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp3_lp_gains_seed1/eval.gif){group="exp3"}

![CPT-PG](outputs/cpt-pg_exp3_lp_gains_seed1/eval.gif){group="exp3"}

![GPT-5-mini](outputs/llm_exp3_lp_gains_seed1/eval.gif){group="exp3"}
:::

CPT overweights small probabilities of large gains. Where REINFORCE might hedge and scatter across paths, CPT-PG commits more aggressively to moderately risky paths, achieving a higher success rate. This is the same bias that makes people buy lottery tickets despite negative expected value.

**Result**: REINFORCE is scattered and hesitant (54% success rate). CPT-PG is more committed to its chosen path (74% success rate). *p* = 1.12 x 10^-5^, *d* = 0.83.


### Experiment 3: Loss Aversion

<starting gif frame>
<main rewards and params table>

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp5_loss_aversion_seed1/eval.gif){group="exp5"}

![CPT-PG](outputs/cpt-pg_exp5_loss_aversion_seed1/eval.gif){group="exp5"}

![GPT-5-mini](outputs/llm_exp5_loss_aversion_seed1/eval.gif){group="exp5"}
:::

This directly tests Kahneman and Tversky's most famous finding: losses hurt roughly twice as much as equivalent gains feel good. A control agent with lambda = 1.0 (no loss aversion) matches REINFORCE, proving that lambda is the causal driver of the behavioral shift.

**Result**: REINFORCE takes the moderate path (mean row 0.99). CPT-PG shifts toward the safest path (mean row 0.80, with 21% of episodes taking row 0). *p* = 9.63 x 10^-29^, *d* = 0.55.

### Experiment 4: The Insurance Policy (d = 0.38)

<starting gif frame>
<main rewards and params table>

::: {layout-ncol=3}
![REINFORCE](outputs/reinforce_exp4_lp_losses_seed1/eval.gif){group="exp4"}

![CPT-PG](outputs/cpt-pg_exp4_lp_losses_seed1/eval.gif){group="exp4"}

![GPT-5-mini](outputs/llm_exp4_lp_losses_seed1/eval.gif){group="exp4"}
:::

CPT overweights small probabilities of catastrophic losses --- like someone buying insurance against a rare disaster. The 0.5% risk on the efficient path *feels* much larger than its true probability. This is the same bias that makes people pay hundreds of dollars a year to insure a phone they are unlikely to break.

**Result**: REINFORCE takes the efficient path (mean row 1.52). CPT-PG shifts safer (mean row 1.26). *p* = 0.0099, *d* = 0.38.

## Technical Description

### Custom Cliff Walking

The environment is a resizable grid world built on top of Gymnasium's CliffWalking. The agent starts at the bottom-left and must reach the bottom-right (the goal). The bottom edge of the grid is a cliff: falling off ends the episode with a configurable penalty. Wind is implemented as a stochastic perturbation --- at each step, with probability `wind_prob`, the agent is pushed one row downward toward the cliff. Higher rows are safer (farther from the cliff edge) but require more steps. Rewards for stepping, reaching the goal, and falling off the cliff are all independently configurable, which lets us construct pure gains domains (positive goal, zero step cost), pure losses domains (negative step cost, negative cliff penalty), and mixed domains. Different configurations of grid size, wind strength, and reward structure create qualitatively different risk profiles, each designed to isolate a specific CPT effect.

### REINFORCE

REINFORCE is a Monte Carlo policy gradient algorithm. The agent runs complete episodes, computes discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$, and updates its policy by ascending the gradient $\nabla_\theta J = \mathbb{E}[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]$. An exponential moving average (EMA) baseline subtracts the running mean return to reduce variance without introducing bias. Entropy regularization $H[\pi]$ is added to the loss and annealed from a high initial coefficient to a low final value, encouraging exploration early in training and convergence later. The policy network is a two-layer MLP mapping one-hot state encodings to action probabilities via softmax. This is the "rational" agent: it converges to the policy that maximizes expected discounted return.

### Cumulative Prospect Theory

Cumulative Prospect Theory models four systematic deviations from rational decision-making. First, *diminishing sensitivity*: the value function $v(x)$ is concave for gains and convex for losses (S-shaped), meaning each additional dollar of gain matters less ($v(x) = x^\alpha$ for gains, $v(x) = -\lambda|x|^\beta$ for losses). Second, *loss aversion*: losses are amplified by a factor $\lambda \approx 2.25$, so losing \$100 feels as bad as gaining \$225 feels good. Third, *probability weighting*: small probabilities are overweighted and large probabilities are underweighted via the weighting function $w(p) = p^\gamma / (p^\gamma + (1-p)^\gamma)^{1/\gamma}$, which is why people simultaneously buy lottery tickets (overweighting small gains) and insurance (overweighting small losses). Fourth, *reference dependence*: outcomes are evaluated relative to a reference point, not in absolute terms. Together, these four components predict the "fourfold pattern" of risk attitudes: risk-averse for likely gains, risk-seeking for unlikely gains, risk-seeking for likely losses, and risk-averse for unlikely losses.

### CPT-PG: Making REINFORCE Human

CPT-PG (Lepel & Barakat, 2024) replaces REINFORCE's raw returns $G$ with CPT-distorted weights $\hat{\varphi}$ computed via the Choquet integral over the batch of trajectories. For each episode $i$ in a batch, the algorithm computes the total discounted return $R_i$, applies the CPT value function $v(R_i)$ to get gains $u^+ = \max(v(R_i), 0)$ and losses $u^- = \max(-v(R_i), 0)$, then integrates against the empirical survival function weighted by $w'(\hat{S}(z))$ to obtain $\hat{\varphi}(R_i) = \int_0^{u^+} w'_+(\hat{S}_+(z))\,dz - \int_0^{u^-} w'_-(\hat{S}_-(z))\,dz$. This scalar $\hat{\varphi}$ replaces $G_t - b$ in the policy gradient. In pure domains (all gains or all losses), $\hat{\varphi}$ values are all same-signed and near-uniform, collapsing the algorithm back to vanilla REINFORCE. The `center_phi` trick subtracts the batch mean $\hat{\varphi}$ before using it as a weight, creating relative positive and negative weights that restore the CPT-induced preference ordering.

### Other Experiments

#### Per-Step CPT Agent

Instead of computing CPT weights over whole episodes, this variant applies CPT probability weighting at each timestep. At every step $t$, the distribution of $G_t$ values across episodes in the batch defines a prospect, and CPT decision weights $\pi_{i,t}$ are computed from this distribution. A sliding window with importance-sampling correction accumulates statistics across batches for stability. While theoretically appealing --- it respects the per-step structure of the MDP --- this approach has higher variance because each timestep has fewer samples than the full episode. In practice, it converged more slowly and less reliably than CPT-PG.

#### CPT-PG + RUDDER

This variant combines CPT-PG with RUDDER-style learned credit assignment. A per-step MLP is trained to decompose the episode-level $\hat{\varphi}$ into per-step contributions $\tilde{\varphi}_t$, with a residual correction to enforce return equivalence ($\sum_t \tilde{\varphi}_t = \hat{\varphi}$). The idea is to focus the policy gradient on causally relevant actions rather than weighting all timesteps equally. While the direction is promising --- it should improve credit assignment in long episodes --- training the RUDDER model added instability and did not outperform standard CPT-PG in our experiments.


## Conclusions and Lessons Learned

Four experiments confirmed that CPT-PG produces the behavioral shifts predicted by Cumulative Prospect Theory. The "Sure Thing" effect (Exp 1, *d* = 0.88) and "Lottery Ticket" effect (Exp 3, *d* = 0.83) both show large effect sizes, while loss aversion (Exp 5, *d* = 0.55) and the "Insurance Policy" (Exp 4, *d* = 0.38) show moderate effects with high statistical significance.

Key technical insights from this project:

- **The `center_phi` trick is essential.** In pure domains (all gains or all losses), CPT weights are all same-signed, reducing CPT-PG to vanilla REINFORCE. Subtracting the batch mean restores the CPT effect.
- **Domain matters.** CPT-PG works reliably in losses domains and pure positive gains. It fails in mixed domains (positive goal, negative cliff, zero step cost) because sparse rewards combined with `center_phi` can produce zero gradients.
- **Lambda cancels in pure domains.** Loss aversion only matters when outcomes straddle the reference point. In pure losses, lambda is just a scaling factor. A shifted reference point activates lambda without leaving the losses training domain.
- **Simpler is better.** Per-step CPT and RUDDER decomposition added complexity without clear gains. The standard CPT-PG algorithm with `center_phi` was the most reliable approach.

The LLM agent (GPT-5-mini) makes for an interesting comparison point. It receives the same environment information but has no training --- just a system prompt describing the grid. Its behavior varies across experiments but generally reflects a "common sense" risk attitude that sometimes aligns with CPT predictions and sometimes does not.


## References

- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5(4), 297-323.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-292.
- Lepel, T., & Barakat, A. (2024). CPT-PG: Cumulative Prospect Theory in Policy Gradient. *arXiv:2410.02605*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229-256.
- Arjona-Medina, J. A., et al. (2019). RUDDER: Return Decomposition for Delayed Rewards. *NeurIPS 2019*.
