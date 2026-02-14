# CPT vs EV Experiment Execution Plan

This document is the operating manual for a team of agents executing experiments that compare rational (Expected Value / REINFORCE) and descriptive (Cumulative Prospect Theory / CPT-PG) decision-making in a custom Cliff Walking environment.

---

## Project Goal

Find measurable behavioral differences between a rational REINFORCE agent and a human-like CPT-PG agent. The primary signal is **which row (path) the agent chooses to traverse** — some rows are riskier (closer to cliff), some rows are safer.

We need **at least 3 replicable differences found**. Focus on finding at least three significant differences first, then focus on the rest of experiments.

* THE SUCCESS CRITERIA IS THAT THE CPT-PG AND REINFORCE AGENT CONVERGE TO DIFFERENT MAX TRAVERSAL ROW IN THE AVERAGE SCENARIO, WHICH MUST BE CORRELATED WITH WHAT THE ACTUAL CPT THEORY EXPECTS. This should be visible in the gifs saved.

---

## Team Structure

### Overview

```
lead (team lead, orchestrator)
├── researcher-1 (experiment executor)
├── researcher-2 (experiment executor)
└── researcher-3 (experiment executor)
```

The team uses a **pool-based experiment assignment** model. The lead maintains the experiment queue and assigns work. Researcher agents pick up the next unfinished experiment when they complete their current one. The lead **actively controls parallel execution** to manage memory constraints — never let all three researchers run heavy training simultaneously. Keep agents busy with a mix of computation-heavy (training) and computation-light (analytical search, paper review, config design) tasks at all times.

### Agent Roster

| Agent Name | Role | Subagent Type | Key Tools |
|---|---|---|---|
| `lead` | Orchestrator, resource coordinator | general-purpose | Task management, messaging, resource monitoring |
| `researcher-1` | Experiment executor | general-purpose | Bash (training runs), file read/write (configs), analytical scripts |
| `researcher-2` | Experiment executor | general-purpose | Bash (training runs), file read/write (configs), analytical scripts |
| `researcher-3` | Experiment executor | general-purpose | Bash (training runs), file read/write (configs), analytical scripts |

### Experiment Queue (priority order)

| Priority | Experiment | Status | Assigned To |
|---|---|---|---|
| 1 | Exp 1: Risk Aversion for High-Probability Gains | Pending | — |
| 2 | Exp 2: Risk Seeking for High-Probability Losses | Pending | — |
| 3 | Exp 3: Risk Seeking for Low-Probability Gains | Pending | — |
| 4 | Exp 4: Risk Aversion for Low-Probability Losses | Pending | — |
| 5 | Exp 5: Loss Aversion (mixed domain) | Pending | — |
| 6 | Exp 6: Allais Paradox (optional) | Pending | — |
| 7 | Exp 7: Status Quo Bias (optional) | Pending | — |
| 8 | Exp 8: Endowment Effect (optional) | Pending | — |

Experiments 1-5 are **required**. Experiments 6-8 are **stretch goals** — only start them after all required experiments have confirmed results.

---

## Agent Role Definitions

### `lead` — Team Lead and Orchestrator

**Identity**: You are the team lead. You coordinate all agents, manage compute resources, and ensure experiments run smoothly. You do NOT run experiments yourself.

**Responsibilities**:

1. **Startup verification**: Before any experiments begin, verify codebase changes are applied (deep merge in `utils.py`, output directory naming in `main.py` — these should already be done).
2. **Experiment assignment**: Assign experiments from the queue to idle researchers. Use the task list to track assignments. When a researcher finishes an experiment (regardless of outcome), assign them the next unfinished experiment from the queue.
3. **Resource coordination**: Enforce the **maximum 2 concurrent training runs** constraint. Analytical config search scripts are lightweight and do NOT count toward this limit — they can always run alongside training.
4. **Memory-aware scheduling**: You have 3 researchers but must **stagger their heavy work** to avoid memory pressure. At any given time:
   - At most 2 researchers should be running training (heavy compute).
   - The 3rd researcher should be doing lightweight work: analytical config search, reading papers/code, designing configs, reviewing results.
   - Rotate who is doing heavy vs. light work as experiments progress.
   - **Keep all agents busy at all times** — if a researcher can't train, assign them preparatory work (config search, paper review, result analysis) for their next experiment.
5. **Initial scheduling**:
   - Assign Exp 1 to `researcher-1` (training) and Exp 2 to `researcher-2` (training).
   - Assign `researcher-3` to begin analytical config search and paper review for Exp 3 (lightweight prep).
   - When a training slot opens, `researcher-3` can start training and the finished researcher moves to lightweight prep for their next experiment.
6. **Monitoring**: Periodically check on researcher progress. If a researcher is stuck (3+ failed config iterations with no behavioral difference), help them brainstorm alternative parameter ranges or escalate.
7. **Result validation**: Each researcher validates their own results — they run the confirmation (4-seed) run and statistical analysis themselves. If validation fails, they iterate immediately without waiting for reassignment.
8. **Final report**: After all required experiments are validated, aggregate results into a summary.

**Communication protocol**:
- Researchers message you when: starting a run, completing a run, finding a promising result, or getting stuck.
- You message researchers when: assigning new experiments, providing parameter suggestions, or flagging resource conflicts.

**Decision authority**: You approve or reject experiment reassignments, resolve resource conflicts between researchers, and decide when to move from required to stretch experiments.

---

### `researcher-1` / `researcher-2` / `researcher-3` — Experiment Executors

**Identity**: You are a researcher. You execute experiments end-to-end: analytical config search, config creation, training runs, and initial result assessment. You are also expected to **deeply understand the theoretical foundations** by reading the research papers and codebase. When you finish one experiment, you report results and pick up the next assignment from the lead.

**Research foundation**: Before and during experimentation, you should read and reference:
- `research/reinforce.pdf` — REINFORCE algorithm foundations
- `research/cpt-pg.pdf` — CPT-PG algorithm (the core paper for this project)
- `research/prospect_theory.pdf` — Kahneman & Tversky's original prospect theory
- `research/cumulative_prospect_theory.pdf` — Tversky & Kahneman's cumulative prospect theory extension

Use these papers to guide your experimentation. Understand *why* certain parameter choices should produce specific behaviors. When iterating on configs, reason from the theory — don't just grid-search blindly. Your goal is to find configs that produce the behavioral differences predicted by CPT. If results don't align with theory and research, investigate why before moving on.

Also read the codebase (`agents.py`, `utils.py`, `custom_cliff_walking.py`, `path_likelihood.py`) to understand how the theory is implemented in practice.

**Responsibilities**:

1. **Analytical divergence search** (Phase 1): Adapt `scripts/find_divergent_config.py` for your assigned experiment. Use `path_likelihood.py` to find configs where EV and CPT analytically prefer different paths in the direction predicted by the experiment hypothesis. This is lightweight and can always run.
2. **Tipping point search** (Phase 2a): Starting from the Phase 1 config, binary-search the primary parameter (gamma for gains domain, reward_step/wind_prob for losses domain) to find where REINFORCE is near-indifferent between the risky and safe paths (EV margin < 5% between top-2 rows). See the "Tipping Point Search Reference" section.
3. **REINFORCE-only validation** (Phase 2b): Run REINFORCE alone (`agents: [reinforce]`, 2 seeds) to confirm the tipping point — look for slow convergence, reward oscillation, and path distribution spread. **Message the lead before starting.** If not at tipping point, adjust and repeat Phase 2a.
4. **CPT-PG training** (Phase 3): Run both agents on the tipping-point config (`agents: [reinforce, cpt-pg]`, 2 seeds). Since REINFORCE is near the decision boundary, CPT's distortions should push CPT-PG to the other side. **Message the lead before starting.**
5. **Theory-guided diagnosis** (Phase 4): If no behavioral divergence, use the diagnostic checklist in the "Tipping Point Search Reference" section to identify which CPT mechanism is inactive or misdirected. Adjust parameters and loop back to Phase 2 or 3. If stuck after 3+ iterations, message the lead.
6. **Self-validation** (Phase 5): When you find a promising 2-seed divergence, validate it yourself — update to `n_seeds: 4`, re-run, pool eval episodes (4 seeds x 20 episodes = 80 datapoints), and compute statistical analysis (mean row, path distribution, Mann-Whitney U test). If validation fails, **iterate immediately** — go back to Phase 3 or 2.
7. **Statistical analysis and report** (Phase 6): Once validation succeeds, produce a structured report (hypothesis confirmed/rejected/inconclusive, effect size, statistical significance) and message the lead with the final results.
8. **Next experiment**: After completing a validated experiment, message the lead to request your next assignment.

**Communication protocol**:
- Message `lead` when: requesting a training slot, reporting final validated results (success or failure), getting stuck, or requesting next assignment.
- Message other researchers when: you discover parameter insights that may help their experiment (e.g., "gamma=0.87 worked well for gains domain").

**What you do NOT do**:
- Do not start a new experiment without the lead's assignment.
- Do not run training without notifying the lead first (resource coordination).

---

## Environment Overview

The Cliff Walking grid has `nrows` rows (0=top, nrows-1=bottom). The cliff occupies the bottom row between start (bottom-left) and goal (bottom-right).

```
Row 0 (safest)    [  ] [  ] [  ] [  ] [  ]
Row 1              [  ] [  ] [  ] [  ] [  ]
Row 2 (risky)      [  ] [  ] [  ] [  ] [  ]
Row 3 (cliff row)  [S ] [XX] [XX] [XX] [G ]
                   start  cliff  cliff  goal
```

**Path choice = risk level**: Agent goes UP to chosen row, RIGHT across, then DOWN to goal. Higher rows = fewer steps but closer to cliff.

**Wind** (`wind_prob`): Each step, the action becomes DOWN with probability `wind_prob`. This is the randomness that creates cliff-fall risk.

**Key probability formulas**:
- Row d=1 (adjacent to cliff): P(cliff) = 1 - (1-wind)^(ncols-1)
- Row d=2: P(cliff) ≈ (ncols-2) * wind^2
- Row d=3: P(cliff) ≈ (ncols-3) * wind^3

---

## How to Run Experiments

```bash
# From the project root: posts/rl_cpt/
python main.py -c config_name
```

This trains all agents listed in `configs/config_name.yaml`, evaluates them, and outputs:
- `outputs/{agent_name}_{config_name}_{seed}/eval.gif` — evaluation episode visualization
- `outputs/{agent_name}_{config_name}_{seed}/training_curves.png` — training plots
- `outputs/{agent_name}_{config_name}_{seed}/history.npz` — raw training data
- Path analysis printed to stdout (row percentages, success rate, cliff rate)

---

## Configuration Reference

### CRITICAL: `stochasticity` must be `"windy"` for wind_prob to take effect!

The `make_env()` function in `custom_cliff_walking.py` only applies wind when `stochasticity == "windy"`. If you forget this, wind_prob will be ignored and all paths become deterministic.

### Parameter Reference

#### Environment Parameters
| Parameter | Controls | Values |
|---|---|---|
| `shape: [nrows, ncols]` | Grid dimensions | 4-5 rows, 5-8 cols recommended |
| `reward_step` | Per-step reward | 0 for positive domain, -1 to -3 for negative domain |
| `reward_cliff` | Cliff fall reward | Negative for loss domain, small positive for gain domain |
| `reward_goal` | Goal arrival reward | Large positive for gain domain, -1 for loss domain |
| `wind_prob` | P(action→DOWN) | 0.03-0.10 = high prob outcomes; 0.20-0.35 = low prob outcomes |
| `stochasticity` | Wind enabler | **Must be `"windy"`** for wind to work |

#### Training Parameters
| Parameter | Controls | Guidance |
|---|---|---|
| `timesteps` | Total env steps | 350k small grids, 500k+ larger/harder |
| `batch_size` | Episodes per update | 8 default, 16-32 for high variance |
| `entropy_coef` / `entropy_coef_final` | Exploration | 0.5→0.01 default |
| `n_seeds` | Reproducibility | **2 for exploration, 4 for confirmation** |
| `n_eval_episodes` | Eval episodes | **20 for all experiments** |

#### Agent Parameters
| Parameter | Controls | Effect |
|---|---|---|
| `lr` | Learning rate | 0.0001 default, 0.0005-0.001 for small grids |
| `gamma` | Discount factor | **Critical in positive domain**. Lower = penalizes longer paths more. 0.80-0.99. |
| `baseline_type` | Variance reduction | `ema` for REINFORCE. CPT-PG uses zero internally. |
| `alpha` | CPT gains exponent | 0.88 (x^0.88, concave for gains) |
| `beta` | CPT losses exponent | 0.88 (-λ|x|^0.88, convex for losses) |
| `lambda_` | Loss aversion | 2.25 (losses feel 2.25x worse). Key for Exp 5. |
| `reference_point` | Gain/loss boundary | 0.0 default. Key for Exp 5 (mixed domain). |
| `w_plus_gamma` | Prob weighting (gains) | 0.61 (overweights small p, underweights large p) |
| `w_minus_gamma` | Prob weighting (losses) | 0.69 (same inverse-S for losses) |
| `center_phi` | Center phi-hat values | **Must be `true` for pure-domain experiments (Exp 1-4)**. Subtracts mean phi so CPT creates relative positive/negative weights. |

---

## Tipping Point Search Reference

### Core Idea

Instead of searching for configs where EV and CPT diverge widely and hoping the RL agents reproduce the divergence, we find REINFORCE's **decision boundary** — the configuration where REINFORCE is nearly indifferent between the risky and safe paths — then train CPT-PG on that borderline config. Since REINFORCE is barely favoring one path, CPT's probability weighting and value distortions only need a small push to flip CPT-PG's preference to the other side.

### Tipping Point Search Algorithm

**For gains domain (Exp 1, 3):**
1. Fix shape, wind_prob, goal_reward, cliff_reward from the Phase 1 config
2. Sweep gamma in [0.80, 0.95] at 0.01 increments
3. For each gamma, compute discounted EV per row using `discounted_ev()` from `scripts/find_divergent_config.py`
4. Record which row EV prefers and the margin: `(EV_best - EV_second) / |EV_best|`
5. Find `gamma_tipping` where the margin is minimized (or where preferred row flips)
6. Set gamma slightly toward the EV-optimal side: target EV margin of 1-5%

**For losses domain (Exp 2, 4):**
1. Fix shape, gamma=0.98, reward_cliff from the Phase 1 config
2. Sweep reward_step in [-0.5, -3.0] and wind_prob in [0.03, 0.10]
3. For each (reward_step, wind_prob), compute EV per row
4. Pick the combination where EV margin is 1-5% between top-2 rows

**For mixed domain (Exp 5):**
1. Keep all rewards in losses domain (step<0, goal<0, cliff<0) for training stability
2. Sweep reference_point to find where some returns fall above and some below the reference
3. Lambda=2.25 amplifies losses relative to gains around the reference point, creating CPT divergence

### How to Recognize the Tipping Point (Training Signatures)

When validating the tipping point with a REINFORCE-only run (Phase 2b), look for these signatures:

| Signature | What to Look For | Too Far from Tipping Point |
|---|---|---|
| **Slow convergence** | Reward doesn't stabilize until 50-80% of timesteps | Converges before 40% of timesteps |
| **Reward oscillation** | Smoothed reward curve (in `rewards.png`) shows up-down swings before settling | Monotonic climb to plateau |
| **Path distribution spread** | eval shows 60-70% on preferred row, 20-30% on alternative | 95%+ on one row |
| **Seed sensitivity** | At least 1 of 2 seeds shows meaningful exploration of alternative path | Both seeds converge instantly to same row |

If REINFORCE converges too quickly or too strongly to one path, the EV advantage is too large — adjust the primary search variable to bring it closer to indifference.

### Probability Weighting Reference Table

Pre-computed values from `CPTWeightingFunction` (`utils.py`). Use this to quickly check whether your experiment's cliff probability falls in the overweighting or underweighting zone.

| p (actual) | w+(p) | ratio w+/p | w-(p) | ratio w-/p | Zone |
|---|---|---|---|---|---|
| 0.01 | 0.034 | 3.4x | 0.021 | 2.1x | Strong overweight |
| 0.05 | 0.103 | 2.1x | 0.079 | 1.6x | Strong overweight |
| 0.10 | 0.166 | 1.7x | 0.135 | 1.4x | Moderate overweight |
| 0.30 | 0.348 | 1.2x | 0.312 | 1.0x | Near neutral |
| 0.50 | 0.421 | 0.84x | 0.435 | 0.87x | Moderate underweight |
| 0.70 | 0.534 | 0.76x | — | — | Strong underweight |
| 0.90 | 0.742 | 0.82x | 0.790 | 0.88x | Strong underweight |

### Theory-Guided Diagnostic Checklist (Phase 4)

When CPT-PG and REINFORCE do NOT diverge, systematically check each CPT mechanism:

| CPT Mechanism | When Active | How to Check | Fix if Inactive |
|---|---|---|---|
| **Value compression** (α=0.88) | Gains > 10 | Is `v(G_risky)/v(G_safe)` significantly different from `G_risky/G_safe`? | Increase goal_reward or cliff_reward to get larger absolute returns |
| **Loss convexity** (β=0.88) | Losses < -10 | Same check for losses | Increase \|step\| or \|cliff\| |
| **Prob overweighting** | p < 0.15 | Check `w(p_cliff)/p_cliff` in table above — is ratio > 1.5? | Adjust wind_prob to put cliff probability in the overweighting zone |
| **Prob underweighting** | p > 0.40 | Check `w(p)/p` in table — is ratio < 0.95? | Adjust wind_prob |
| **Loss aversion** (λ=2.25) | Mixed domain (gains AND losses) | Are returns BOTH above AND below reference_point? | Adjust reference_point to straddle returns. Lambda cancels in pure domains! |
| **center_phi** | Pure domains (all gains or all losses) | Are phi values near-uniform before centering? | Set `center_phi: true` — **required for Exp 1-4** |
| **Tipping point shifted** | After any parameter change | Is REINFORCE still borderline? | Re-run Phase 2a/2b after adjustments |
| **Training failure** | CPT-PG reward curve is flat | Check training curves, compare to REINFORCE | Adjust lr, entropy_coef, batch_size |

---

## Metrics Framework

### Primary: Row Traversal Preference
Measured by `evaluate_paths()` in `utils.py`. For each eval episode, tracks the minimum row reached (the "path" chosen).

- **Mean traversal row**: Lower = safer. Higher = riskier.
- **Path distribution**: % of episodes at each row.
- **Row preference divergence**: Difference in mean traversal row between REINFORCE and CPT-PG.

THE SUCCESS CRITERIA IS THAT THE CPT-PG AND REINFORCE AGENT CONVERGE TO DIFFERENT MAX TRAVERSAL ROW IN THE AVERAGE SCENARIO, WHICH MUST BE CORRELATED WITH WHAT THE ACTUAL CPT THEORY EXPECTS.

### Secondary Metrics
| Metric | How to Measure | Purpose |
|---|---|---|
| Success rate | Episodes reaching goal / total | Performance |
| Cliff fall rate | Episodes falling / total | Direct risk |
| Path consistency | Std dev of min_row | Policy commitment |
| Average episode reward | Mean undiscounted reward | Outcome quality |

### What Counts as a Successful Experiment
1. **Direction matches hypothesis**: CPT prefers safer/riskier paths as predicted
2. **Magnitude**: >0.5 row difference in mean traversal row, or >15% shift in path distribution
3. **Consistency across seeds**: All seeds show same direction
4. **Statistical power**: 2 seeds × 20 episodes = 40 datapoints per agent minimum

---

## Experiment Lifecycle

Each experiment is owned end-to-end by a single researcher. The core strategy is the **tipping-point approach**: find the configuration where REINFORCE is nearly indifferent between paths, then exploit that indifference with CPT-PG. If validation fails, the researcher iterates immediately. The starting configs in each experiment section are initial guesses — researchers WILL need to explore and adjust parameters.

We need **at least 3 replicable differences found**. Focus on finding at least three significant differences first, then focus on the rest of experiments.

### Researcher Phases

#### Phase 1: Analytical Divergence Search (low resource, always allowed)

Adapt `scripts/find_divergent_config.py` to your experiment. Use `path_likelihood.py` functions:
- `cliff_fall_probability(row, nrows, ncols, wind_prob)` — P(cliff) for a row
- `calculate_path_cpt_value(outcomes, value_func, weighting_func)` — CPT value with proper decision weights
- `calculate_path_expected_value(outcomes)` — EV value
- `build_path_outcome_distributions(env_config)` — outcome distributions for each row
- `compare_value_frameworks(env_config, cpt_params)` — compare EV vs CPT preferences (**always pass `use_probability_weighting: True` in cpt_params**)

Search the parameter space listed in your experiment section. Find configs where EV and CPT **diverge in the predicted direction**. The acceptance criterion is: `ev_preferred_row != cpt_preferred_row`, matching the experiment's hypothesis.

#### Phase 2a: Tipping Point Search (analytical, low resource)

Starting from the Phase 1 config, find where REINFORCE is nearly indifferent between the risky and safe paths:

1. Identify the **primary search variable** for your experiment (see per-experiment sections below — typically gamma for gains domain, reward_step/wind_prob for losses domain)
2. Binary-search that variable to find where EV margin between the top-2 rows is < 5%
3. See the **Tipping Point Search Reference** section for the algorithm and domain-specific guidance
4. Create config YAML in `configs/` with `n_seeds: 2` and `agents: [reinforce]` (REINFORCE only for Phase 2b)

#### Phase 2b: Validate the Tipping Point (REINFORCE-only training)

1. **Message the lead** to request a training slot
2. Run: `python main.py -c your_config_name` (REINFORCE only)
3. Check training signatures against the tipping-point criteria (see reference section):
   - Slow convergence (settles at 50-80% of timesteps, not before 40%)
   - Reward oscillation in `rewards.png` before stabilizing
   - Path distribution spread: 60-70% on preferred row, not 95%+ on one row
   - Seed sensitivity: at least 1 of 2 seeds shows exploration of alternative path
4. **If NOT at tipping point**: adjust the primary search variable and repeat Phase 2a
5. **If at tipping point**: proceed to Phase 3

#### Phase 3: CPT-PG Training (2 seeds, resource-constrained)

1. Update config to include both agents: `agents: [reinforce, cpt-pg]`
2. Ensure `center_phi: true` is set for pure-domain experiments (Exp 1-4)
3. **Message the lead** to request a training slot
4. Run: `python main.py -c your_config_name`
5. Check stdout for path analysis, training curves, and eval GIFs
6. Assess: do the agents converge to **different preferred rows** in the direction predicted by CPT theory?

THE SUCCESS CRITERIA IS THAT THE CPT-PG AND REINFORCE AGENT CONVERGE TO DIFFERENT MAX TRAVERSAL ROW IN THE AVERAGE SCENARIO, WHICH MUST BE CORRELATED WITH WHAT THE ACTUAL CPT THEORY EXPECTS.

#### Phase 4: Theory-Guided Diagnosis (if no divergence)

If CPT-PG and REINFORCE choose the same path, use the **diagnostic checklist** in the Tipping Point Search Reference section. Systematically check:

1. **phi-hat uniformity** — need `center_phi: true` in pure domains?
2. **Probability weighting direction** — is w(p)/p pushing in the expected direction for this wind_prob? Check the reference table.
3. **Value compression magnitude** — are absolute returns large enough (> 50) for meaningful compression?
4. **Domain compatibility** — mixed gains domain (cliff<0, goal>0, step=0) fails with CPT-PG. Avoid it.
5. **Tipping point shifted** — did parameter adjustments move the tipping point? Re-verify REINFORCE is still borderline.
6. **Training convergence** — if CPT-PG reward curve is flat, the issue is training not CPT theory. Adjust lr, entropy, batch_size.

After diagnosis, loop back to Phase 2 or 3 with adjusted parameters. After 3+ failed iterations, message the lead for guidance.

**If promising divergence found** — proceed to Phase 5.

### Self-Validation Phases (same researcher)

#### Phase 5: Validation and Confirmation (4 seeds)

1. Review your own 2-seed results critically (path analysis, training curves, eval GIFs)
2. If results look weak or inconsistent, go back to Phase 3 immediately — adjust parameters and re-run
3. If results look promising:
   a. Update the config: `n_seeds: 4`
   b. **Message the lead** to request a training slot
   c. Run: `python main.py -c config_name`
   d. Pool all eval episodes across seeds (4 seeds x 20 episodes = 80 datapoints per agent)
4. **If validation fails** (4-seed results are weaker, inconsistent, or don't match hypothesis): go back to Phase 3 and iterate. Do NOT report a failed validation as final — fix it first.

#### Phase 6: Statistical Analysis and Report

Once 4-seed validation succeeds, produce:
1. Final config (YAML)
2. Path distribution comparison (table with % at each row per agent)
3. All metrics with standard errors (mean row, success rate, cliff rate, episode reward)
4. Statistical significance test (Mann-Whitney U or similar)
5. Hypothesis confirmed / rejected / inconclusive + reasoning
6. Effect size and practical significance
7. Any surprises or insights

### Experiment Flow Diagram

```
Researcher                              Lead
    |                                     |
    |-- Phase 1: Analytical Divergence    |
    |   (use path_likelihood.py)          |
    |   Output: config where EV!=CPT      |
    |                                     |
    |-- Phase 2a: Tipping Point Search    |
    |   (binary search on gamma/step/wind)|
    |   Output: config where EV margin<5% |
    |                                     |
    |-- "need train slot (REINFORCE only)"|
    |   --------------------------------> |
    |                                     |-- "slot approved"
    |-- Phase 2b: Validate Tipping Point  |
    |   (2-seed REINFORCE-only run)       |
    |   Check: slow convergence?          |
    |   Check: path distribution spread?  |
    |   Check: seed sensitivity?          |
    |                                     |
    |   [If NOT at tipping point:         |
    |    adjust, loop to Phase 2a]        |
    |                                     |
    |-- "need train slot (both agents)"   |
    |   --------------------------------> |
    |                                     |-- "slot approved"
    |-- Phase 3: CPT-PG Training          |
    |   (2-seed, both agents)             |
    |   Check: different row preferences? |
    |   Check: matches CPT theory?        |
    |                                     |
    |   [If no divergence: Phase 4        |
    |    diagnosis, loop to Phase 2 or 3] |
    |                                     |
    |-- Phase 5: Validation (4 seeds)     |
    |   [If fails: loop to Phase 3]       |
    |                                     |
    |-- Phase 6: Stats & Report           |
    |-- "final report" -----------------> |
    |-- "request next exp" -------------> |
    |                                     |
```

---

## Experiment 1: Risk Aversion for High-Probability Gains (REQUIRED)

**Assigned to**: Next available researcher (initially `researcher-1`)
**Priority**: 1

### Theory

CPT's fourfold pattern predicts risk aversion when gains are probable:
- Value function concavity (`x^0.88`) compresses large gains
- Probability weighting underweights high probabilities: `w+(0.85) ≈ 0.74 < 0.85`
- Combined: the "sure thing" (safe path) becomes relatively more attractive

### Hypothesis

In a positive-reward domain with low wind, CPT should prefer safer paths than REINFORCE.

### Starting Config (`configs/exp1_hp_gains.yaml`)

```yaml
env:
  shape: [4, 5]
  stochasticity: windy
  reward_cliff: 5
  reward_step: 0
  reward_goal: 100
  wind_prob: 0.05
training:
  timesteps: 350000
  n_eval_episodes: 20
  batch_size: 8
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 2
agent_config:
  lr: 0.001
  gamma: 0.90
  baseline_type: ema
  alpha: 0.88
  beta: 0.88
  lambda_: 2.25
  reference_point: 0.0
  w_plus_gamma: 0.61
  w_minus_gamma: 0.69
  sliding_window_size: 5
  sliding_window_decay: 0.8
agents:
  - reinforce
  - cpt-pg
```

**Key design decisions**:
- `reward_step=0` + positive `reward_goal` → gains domain (all returns positive)
- `wind_prob=0.05` → row 2 (d=1) has P(success)=81.5% (HIGH probability)
- `gamma=0.90` → safe path (10 steps): G=38.7, risky path (6 steps): G=59.0. Gamma creates the differentiation.

### Parameters You Can Modify

| Parameter | Range | Why |
|---|---|---|
| `gamma` | [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95] | **Critical lever**. Lower gamma penalizes safe path more (more discounting). |
| `goal_reward` | [50, 100, 150, 200, 500] | Affects the magnitude of the gain. Larger may increase CPT compression effect. |
| `cliff_reward` | [1, 2, 5, 10] | Small positive. Must stay in gains domain. |
| `wind_prob` | [0.03, 0.05, 0.07, 0.10] | Keep ≤0.10 for high-probability regime. |
| `shape` | [4,5], [4,6], [4,7], [5,5], [5,6] | Larger = more path differentiation but harder to train. |

### Tipping Point Strategy

**Primary search variable**: `gamma`
**Tipping range**: [0.85, 0.92]
**How it works**: At low gamma (0.80-0.85), REINFORCE strongly prefers risky path (short path is less discounted). At high gamma (0.93+), REINFORCE prefers safe path (risk of cliff outweighs discount advantage). The tipping point is in between.

1. **Phase 1**: Find a gamma where EV prefers risky but CPT prefers safe (use `compare_value_frameworks` with `use_probability_weighting: True`)
2. **Phase 2a**: Binary-search gamma to find where REINFORCE's EV margin between risky and safe is < 5%
3. **Phase 2b**: Run REINFORCE-only. Look for oscillation between rows and slow convergence
4. **Phase 3**: Add CPT-PG. CPT's value compression (`v(G_risky)^0.88` compresses large risky gains more than safe gains) + probability underweighting of high success probability should tip CPT toward the safer path

**CPT mechanism active here**: Value compression (α=0.88, concave) + probability underweighting of high-p success (w+(0.82) ≈ 0.71 < 0.82)

### Expected Outcome

- **REINFORCE**: Barely prefers risky path (row 2) at the tipping point — slow to converge, oscillates
- **CPT-PG**: Prefers safe path (row 0 or 1) — value compression + probability underweighting makes safe more attractive
- Mean traversal row: CPT < REINFORCE

---

## Experiment 2: Risk Seeking for High-Probability Losses (REQUIRED)

**Assigned to**: Next available researcher (initially `researcher-2`)
**Priority**: 2

### Theory

CPT predicts risk seeking when losses are probable:
- Value function convexity for losses: a certain loss feels worse than a gamble with the same EV
- Probability underweighting: `w-(0.85) < 0.85` reduces perceived likelihood of bad outcome

### Hypothesis

In an all-negative domain with low wind, CPT should prefer riskier paths than REINFORCE.

### Starting Config (`configs/exp2_hp_losses.yaml`)

```yaml
env:
  shape: [4, 6]
  stochasticity: windy
  reward_cliff: -100
  reward_step: -1
  reward_goal: -1
  wind_prob: 0.08
training:
  timesteps: 350000
  n_eval_episodes: 20
  batch_size: 8
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 2
agent_config:
  lr: 0.0005
  gamma: 0.99
  baseline_type: ema
  alpha: 0.88
  beta: 0.88
  lambda_: 2.25
  reference_point: 0.0
  w_plus_gamma: 0.61
  w_minus_gamma: 0.69
  sliding_window_size: 5
  sliding_window_decay: 0.8
agents:
  - reinforce
  - cpt-pg
```

**Key design decisions**:
- All rewards negative → everything is a "loss" relative to reference_point=0
- `gamma=0.99` → accumulates step losses (safe path cost compounds)
- `wind_prob=0.08` → row 2 (d=1): P(fall)=34.1%, P(success)=65.9% (high probability regime)
- Safe path (~12 steps): G≈-12. Risky success (~8 steps): G≈-8. Risky cliff: G≈-104.

### Parameters You Can Modify

| Parameter | Range | Why |
|---|---|---|
| `reward_cliff` | [-50, -80, -100, -150] | Cliff severity. More negative = bigger gamble. |
| `reward_step` | [-0.5, -1, -1.5, -2, -3] | Higher cost makes safe path's "certain loss" worse. |
| `wind_prob` | [0.05, 0.07, 0.08, 0.10] | Keep in high-probability regime. |
| `shape` | [4,5], [4,6], [4,7] | Affects step count differences between paths. |
| `gamma` | [0.95, 0.97, 0.99] | Higher accumulates more step losses. |

### Tipping Point Strategy

**Primary search variable**: `reward_step` (secondary: `wind_prob`)
**Tipping range**: reward_step in [-1.0, -2.5], wind_prob in [0.05, 0.10]
**How it works**: At high |step| (e.g., -2.5), REINFORCE prefers risky path (safe path's many steps are very costly). At low |step| (e.g., -0.5), REINFORCE prefers safe (cliff risk dominates over step cost). The tipping point is in between.

1. **Phase 1**: Find a reward_step where EV prefers safe but CPT prefers risky
2. **Phase 2a**: Sweep reward_step to find where REINFORCE's EV margin is < 5%
3. **Phase 2b**: Run REINFORCE-only. Look for oscillation and slow convergence
4. **Phase 3**: Add CPT-PG. CPT's loss convexity (−λ|x|^0.88 is convex, making certain losses feel disproportionately bad) + probability underweighting (w−(0.66) < 0.66, reduces perceived high success probability) should push CPT toward the riskier path

**CPT mechanism active here**: Loss convexity (β=0.88) + probability underweighting of high p_success. Use `center_phi: true` since all returns are negative.

### Expected Outcome

- **REINFORCE**: Barely prefers safe path at the tipping point — slow to converge
- **CPT-PG**: Prefers risky path (gambles to escape certain loss)
- Mean traversal row: CPT > REINFORCE
- CPT may have higher cliff fall rate (confirms risk-seeking)

---

## Experiment 3: Risk Seeking for Low-Probability Gains (REQUIRED)

**Assigned to**: Next available researcher
**Priority**: 3

### Theory

"Lottery ticket" behavior: CPT overweights small probabilities of large gains.
- `w+(0.10) ≈ 0.18` — nearly doubles the perceived probability
- Despite value compression (x^0.88), the probability overweighting can dominate

### Hypothesis

With high wind and positive rewards, CPT should pursue the risky path more than REINFORCE.

### Starting Config (`configs/exp3_lp_gains.yaml`)

```yaml
env:
  shape: [4, 8]
  stochasticity: windy
  reward_cliff: 2
  reward_step: 0
  reward_goal: 500
  wind_prob: 0.25
training:
  timesteps: 500000
  n_eval_episodes: 20
  batch_size: 16
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 2
agent_config:
  lr: 0.0005
  gamma: 0.92
  baseline_type: ema
  alpha: 0.88
  beta: 0.88
  lambda_: 2.25
  reference_point: 0.0
  w_plus_gamma: 0.61
  w_minus_gamma: 0.69
  sliding_window_size: 5
  sliding_window_decay: 0.8
agents:
  - reinforce
  - cpt-pg
```

**Key design decisions**:
- `wind_prob=0.25` → row 2 (d=1): P(fall)=86.7%, P(success)=13.3% (LOW probability)
- `reward_goal=500` → large "lottery prize"
- `reward_step=0` → gains domain only
- `batch_size=16` → more samples per update for high-variance environment
- `timesteps=500000` → more training for harder environment

### Parameters You Can Modify

| Parameter | Range | Why |
|---|---|---|
| `goal_reward` | [200, 500, 1000, 2000, 5000] | **Must be large enough** for probability overweighting to overcome value compression. |
| `cliff_reward` | [1, 2, 5] | Keep small positive. |
| `wind_prob` | [0.20, 0.25, 0.30, 0.35] | Low-probability regime. |
| `gamma` | [0.85, 0.88, 0.90, 0.92, 0.95] | Balances path length discount. |
| `shape` | [4,7], [4,8], [4,10], [5,7] | Wider = more wind exposure. |
| `batch_size` | [16, 32] | Larger batches stabilize high-variance training. |
| `timesteps` | [500000, 750000] | May need more training for convergence. |

**This is the hardest experiment.** The probability overweighting effect is strong theoretically but may be hard to learn. If the risky path almost never succeeds during training, the agent may never explore it enough.

### Tipping Point Strategy

**Primary search variable**: `gamma` (secondary: `wind_prob`)
**Tipping range**: gamma in [0.85, 0.92], wind_prob in [0.20, 0.35]
**How it works**: At low gamma, REINFORCE prefers risky (short path barely discounted, lottery worth trying). At high gamma, REINFORCE prefers safe (risk-adjusted EV of long safe path dominates). The tipping point is where REINFORCE barely prefers safe.

1. **Phase 1**: Find a gamma/wind_prob where EV prefers safe but CPT prefers risky (probability overweighting flips CPT's preference)
2. **Phase 2a**: Binary-search gamma to find where REINFORCE barely prefers safe (EV margin < 5%)
3. **Phase 2b**: Run REINFORCE-only. Since success is rare, look for: mostly safe path but occasional risky exploration, seed disagreement on preferred row
4. **Phase 3**: Add CPT-PG. CPT overweights the small probability of the big gain: `w+(0.10) ≈ 0.17` (1.7x overweight). This should be enough to flip CPT-PG toward the risky path at the tipping point.

**CPT mechanism active here**: Probability overweighting of small p_success (w+(p) >> p for p < 0.15). Check the probability weighting table for your specific wind_prob.

**Practical tips for this hard experiment:**
- Consider row 1 (d=2) instead of row 2 (d=1) for intermediate risk levels with higher success rates
- Use very large goal rewards (1000+) to amplify the lottery signal
- Use `batch_size: 32` to stabilize the high-variance environment
- May need `timesteps: 500000-750000` for convergence

### Expected Outcome

- **REINFORCE**: Barely prefers safe path at the tipping point — slow convergence, explores risky path occasionally
- **CPT-PG**: Weaker safe preference or risky preference (overweights small p of big gain)
- Mean traversal row: CPT > REINFORCE
- CPT likely has higher cliff fall rate (the "lottery" usually loses)

---

## Experiment 4: Risk Aversion for Low-Probability Losses (REQUIRED)

**Assigned to**: Next available researcher
**Priority**: 4

### Theory

"Insurance" behavior: CPT overweights small probabilities of catastrophic losses.
- `w-(0.014) ≈ 0.035` — 2.5x overweighting
- Combined with `lambda_=2.25` loss aversion: perceived risk ≈ 5.6x worse than EV

### Hypothesis

With low wind and negative rewards, the risky path has a tiny cliff probability that EV ignores but CPT heavily overweights. CPT should be more cautious.

### Starting Config (`configs/exp4_lp_losses.yaml`)

```yaml
env:
  shape: [5, 6]
  stochasticity: windy
  reward_cliff: -100
  reward_step: -1
  reward_goal: -1
  wind_prob: 0.06
training:
  timesteps: 400000
  n_eval_episodes: 20
  batch_size: 8
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 2
agent_config:
  lr: 0.0005
  gamma: 0.99
  baseline_type: ema
  alpha: 0.88
  beta: 0.88
  lambda_: 2.25
  reference_point: 0.0
  w_plus_gamma: 0.61
  w_minus_gamma: 0.69
  sliding_window_size: 5
  sliding_window_decay: 0.8
agents:
  - reinforce
  - cpt-pg
```

**Key design decisions**:
- 5 rows → 4 path options (rows 0-3) with nuanced cliff probabilities
- `wind_prob=0.06` → row 3 (d=1): P(fall)=26.6%; row 2 (d=2): P(fall)≈1.4% (LOW); row 1 (d=3): P(fall)≈0.065% (negligible)
- The key comparison is row 2: EV sees 1.4% cliff as trivial; CPT sees it as ~5.6x worse

### Parameters You Can Modify

| Parameter | Range | Why |
|---|---|---|
| `wind_prob` | [0.04, 0.05, 0.06, 0.07, 0.08] | Keep row 2 cliff probability ~1-3%. |
| `reward_cliff` | [-50, -80, -100, -150, -200] | Amplifies the rare loss. |
| `reward_step` | [-0.5, -1, -1.5, -2] | Step penalty differentiates paths. |
| `shape` | [5,5], [5,6], [5,7], [5,8] | 5 rows for more path granularity. |
| `gamma` | [0.97, 0.98, 0.99] | High gamma in loss domain. |

### Tipping Point Strategy

**Primary search variable**: `wind_prob` (secondary: `reward_step`)
**Tipping range**: wind_prob in [0.04, 0.08], targeting row 2 cliff probability ~1-3%
**How it works**: At low wind_prob, REINFORCE prefers row 2 (tiny risk, shorter path). At higher wind_prob, REINFORCE shifts to row 1 (risk becomes non-negligible). The tipping point is where REINFORCE barely prefers row 2.

1. **Phase 1**: Find a wind_prob where EV prefers row 2 but CPT prefers row 1 (CPT overweights the small cliff probability at row 2)
2. **Phase 2a**: Sweep wind_prob to find where REINFORCE's EV margin between row 2 and row 1 is < 5%
3. **Phase 2b**: Run REINFORCE-only. Look for: some seeds choosing row 2, others choosing row 1
4. **Phase 3**: Add CPT-PG. CPT overweights the small cliff probability: `w-(0.014) ≈ 0.035` (2.5x overweight). This should tip CPT-PG away from row 2 toward the safer row 1 or 0.

**CPT mechanism active here**: Probability overweighting of small p_cliff (w-(p) >> p for p < 0.05). Use `center_phi: true` since all returns are negative.

### Expected Outcome

- **REINFORCE**: Barely accepts row 2 at the tipping point (tiny risk, shorter path, fewer step losses)
- **CPT-PG**: Avoids row 2, prefers row 0 or 1 (overweights the tiny cliff probability)
- Mean traversal row: CPT < REINFORCE
- CPT should have near-0% cliff falls

---

## Experiment 5: Loss Aversion (REQUIRED)

**Assigned to**: Next available researcher
**Priority**: 5

### Theory

Loss aversion (`lambda_=2.25`): losses loom 2.25x larger than equivalent gains. In a mixed prospect where the risky path can produce gains OR losses relative to the reference point, the loss side is amplified.

### Hypothesis

With a reference point creating a mixed gains/losses prospect, CPT should strongly avoid the risky path's potential loss.

### Starting Config (`configs/exp5_loss_aversion.yaml`)

```yaml
env:
  shape: [4, 6]
  stochasticity: windy
  reward_cliff: -50
  reward_step: -1
  reward_goal: 10
  wind_prob: 0.10
training:
  timesteps: 400000
  n_eval_episodes: 20
  batch_size: 8
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 2
agent_config:
  lr: 0.0005
  gamma: 0.99
  baseline_type: ema
  alpha: 0.88
  beta: 0.88
  lambda_: 2.25
  reference_point: -5.0
  w_plus_gamma: 0.61
  w_minus_gamma: 0.69
  sliding_window_size: 5
  sliding_window_decay: 0.8
agents:
  - reinforce
  - cpt-pg
```

**Key design decisions**:
- Mixed domain: `reward_goal=10` (positive) + `reward_cliff=-50` (negative) + `reward_step=-1`
- `reference_point=-5.0` → safe path return (~-12) is a small loss (-7 relative), risky success (~+2) is a gain (+7), risky cliff (~-54) is a large loss (-49 relative)
- `lambda_=2.25` amplifies the -49 by 2.25x in CPT value function

### Parameters You Can Modify

| Parameter | Range | Why |
|---|---|---|
| `reference_point` | [-3, -5, -7, -10, -15] | Must straddle risky path outcomes. |
| `reward_goal` | [5, 10, 15, 20, 30] | Controls gain magnitude. |
| `reward_cliff` | [-30, -50, -80, -100] | Controls loss magnitude. |
| `reward_step` | [-0.5, -1, -1.5] | Background cost. |
| `wind_prob` | [0.08, 0.10, 0.12, 0.15] | Moderate regime. |

### Control Experiment

Also run CPT-PG with `lambda_=1.0` (loss aversion disabled) to confirm the effect is driven by lambda_:

```yaml
agents:
  - reinforce
  - name: cpt-pg
    lambda_: 2.25
  - name: cpt-pg
    lambda_: 1.0
```

The output directory fix ensures these don't collide: `cpt-pg_lambda_2.25_{config}` vs `cpt-pg_lambda_1.0_{config}`.

### Tipping Point Strategy

**Primary search variable**: `reference_point`
**How it works**: This experiment requires a mixed domain where some returns are "gains" and some are "losses" relative to the reference point, so that lambda=2.25 can differentially amplify the loss side. However, mixed gains domain (cliff<0, goal>0, step=0) is known to FAIL with CPT-PG.

**Critical design constraint**: Keep all rewards in losses domain (step<0, goal<0, cliff<0) for training stability. Use a shifted `reference_point` (e.g., -20) so that:
- Safe path returns (e.g., -12) are ABOVE the reference → treated as "gains" by CPT
- Cliff returns (e.g., -104) are BELOW the reference → treated as "losses" by CPT, amplified by λ=2.25

1. **Phase 1**: Find a reference_point where EV is near-indifferent but CPT strongly prefers safe (lambda amplifies the loss side)
2. **Phase 2a**: Sweep reference_point to find where REINFORCE barely prefers risky (EV margin < 5%)
3. **Phase 2b**: Run REINFORCE-only to validate the tipping point
4. **Phase 3**: Add CPT-PG with λ=2.25. The loss amplification should push CPT-PG firmly toward safe

### Expected Outcome

- **REINFORCE**: Barely prefers risky path at the tipping point
- **CPT-PG (λ=2.25)**: Strongly avoids risky path (loss amplified 2.25x)
- **CPT-PG (λ=1.0, control)**: Behaves more like REINFORCE
- The λ=2.25 vs λ=1.0 comparison directly isolates loss aversion's impact

---

## Experiment 6: Allais Paradox Analog (OPTIONAL — Stretch Goal)

**Assigned to**: Next available researcher (stretch goal)
**Priority**: 6

### Theory

The certainty effect: difference between 100% and 99% has far more psychological impact than between 10% and 11%.

### Design

Two sub-configs with the same ~1% probability gap at different base rates:

**Sub-A** (near certainty): `wind_prob=0.003`, shape [5,5] → row 3 has ~1.2% cliff risk. Compare row 0 (certain) vs row 3 (99% safe).

**Sub-B** (low base): `wind_prob=0.30`, shape [4,8] → row 2 has ~87% cliff risk vs row 1 with ~86% risk (~1% gap at low base).

CPT should show a LARGER behavioral shift in Sub-A than Sub-B.

**Note**: This is conceptually interesting but the signal is subtle. Only attempt if experiments 1-5 produce clear results.

---

## Experiment 7: Status Quo Bias (OPTIONAL — Stretch Goal)

**Assigned to**: Next available researcher (stretch goal)
**Priority**: 7

### Theory

CPT predicts resistance to changing learned behavior because departures are evaluated asymmetrically.

### Design

Two-phase training:
1. Train both agents on config A (moderate wind, balanced risk)
2. Switch to config B (high wind → risky path now much worse)

Measure: episodes until agent shifts from learned path to new optimal.

**Code change needed**: Add save/load model weights to `agents.py` for continuity between phases. Implementation:
```python
# In BaseAgent or REINFORCEAgent:
def save_weights(self, path):
    torch.save(self.policy.state_dict(), path)

def load_weights(self, path):
    self.policy.load_state_dict(torch.load(path))
```

---

## Experiment 8: Endowment Effect (OPTIONAL — Stretch Goal)

**Assigned to**: Next available researcher (stretch goal)
**Priority**: 8

### Theory

As the agent performs better, its reference point rises, making any performance decline feel like a loss → increasingly conservative behavior.

### Design

Modify CPT-PG to use adaptive reference point (EMA of recent returns):
```python
# In CPTPGAgent.learn(), after each batch:
if self.adaptive_reference:
    batch_mean = np.mean(batch_returns)
    self.reference_point = 0.99 * self.reference_point + 0.01 * batch_mean
    self.cpt_value.reference_point = self.reference_point
```

Compare CPT-PG (adaptive reference) vs CPT-PG (fixed reference=0) vs REINFORCE.

**Code change needed**: Add `adaptive` reference point mode to CPTPGAgent.

---

## Critical Files Reference

| File | Purpose | When to Read |
|---|---|---|
| `agents.py` | REINFORCEAgent, CPTPGAgent implementations | Before any experiment |
| `utils.py` | `load_config()`, `evaluate_paths()`, CPT utility classes | Before any experiment |
| `custom_cliff_walking.py` | Environment (wind, rewards, termination) | Before any experiment |
| `path_likelihood.py` | Analytical EV/CPT calculations, `cliff_fall_probability()` | During config search |
| `scripts/find_divergent_config.py` | Template for config search scripts | During config search |
| `main.py` | Training loop, evaluation, output directories | When debugging runs |
| `configs/base.yaml` | Default config (all new configs inherit from this) | Always |

---

## Notes

* Remember to use only `reinforce` and `cpt-pg` agents. The others are blocked for now.
* Large batch sizes is pretty important since some important estimators depend on the batch size. Don't have it lower than 32.
* After the experiment is done, save a single gift with the expected behavior for each agent.
* Prepare a good strategy for learning when we need high wind probabilities for the experiment, which will make learning harder (example: taller grids, different reward values, etc.)
* THE SUCCESS CRITERIA IS THAT THE CPT-PG AND REINFORCE AGENT CONVERGE TO DIFFERENT MAX TRAVERSAL ROW IN THE AVERAGE SCENARIO, WHICH MUST BE CORRELATED WITH WHAT THE ACTUAL CPT THEORY EXPECTS. This should be visible in the gifs saved.

## Important Reminders

### Technical
1. **`stochasticity: windy`** — ALWAYS set this in configs that use wind_prob. Without it, wind is disabled.
2. **Positive domain requires `reward_step: 0`** — non-zero step rewards in positive domain cause unwanted behavior. Use gamma as the path-length penalty.
3. **Deep merge is active** — experiment configs only need to specify parameters that differ from base.yaml.
4. **`center_phi: true` is required** for all pure-domain experiments (Exp 1-4). Without it, phi-hat values are near-uniform and CPT-PG behaves like REINFORCE.
5. **`compare_value_frameworks` must use probability weighting** — always pass `use_probability_weighting: True` in cpt_params when calling this function.
6. **REINFORCE-only training runs** (Phase 2b) are half the compute of a full run and do NOT count against the 2-concurrent-run limit.

### Resource Management
7. **2 concurrent training runs maximum** — researchers MUST message the lead before starting any training run. Analytical config search scripts and REINFORCE-only Phase 2b runs do NOT count (they are lightweight / half compute). The lead tracks active slots and approves/denies requests.
8. The lead should **actively monitor compute resources** and preemptively manage the training queue to prevent crashes.

### Experiment Execution
9. **Use the tipping-point strategy** — the starting configs are educated guesses. Follow the lifecycle: analytical divergence search (Phase 1) → tipping point search (Phase 2a) → REINFORCE-only validation (Phase 2b) → CPT-PG training (Phase 3) → diagnosis if needed (Phase 4) → 4-seed validation (Phase 5). If validation fails, iterate immediately.
10. **The main goal is behavioral differences** — a "successful" experiment shows CPT and REINFORCE choosing different paths, in the direction predicted by CPT theory. THE SUCCESS CRITERIA IS THAT THE CPT-PG AND REINFORCE AGENT CONVERGE TO DIFFERENT MAX TRAVERSAL ROW IN THE AVERAGE SCENARIO, WHICH MUST BE CORRELATED WITH WHAT THE ACTUAL CPT THEORY EXPECTS.
11. **Find REINFORCE's indifference point first** — the key insight is that CPT's distortions are small relative to a strong EV preference. By finding where REINFORCE is barely choosing one path, even small CPT effects can flip the preference. Don't skip Phase 2.
12. **Use the diagnostic checklist when stuck** — Phase 4 provides a structured theory-grounded approach to debugging. Check each CPT mechanism against the probability weighting table before making random parameter changes.
13. **Document everything** — record what configs you tried, what worked, what didn't, and why.
14. We need **at least 3 replicable differences found**. Focus on finding at least three significant differences first, then focus on the rest of experiments.

### Team Coordination
15. **Researchers own experiments end-to-end** — each researcher explores, validates, and reports their own experiment. After completing a validated experiment, message the lead for your next assignment. Do not sit idle.
16. **Self-validation keeps momentum** — researchers validate their own results and iterate immediately on failure, avoiding handoff delays. The only resource constraint is the 2-concurrent-run limit.
17. **Any code changes that can break the experiments flow must be coordinated and confirmed with the lead** to avoid catastrophic changes.
18. **Cross-agent communication is encouraged** — share parameter insights, ask questions, and flag issues. Asking questions improves speed and success probability. Use direct messages for targeted info, not broadcasts.
19. Feel free to **read the research, review the codebase, or run calculations as needed**.

**Feel free to ask any questions** you need to clarify or improve experimentation performance. This goes to the lead and all the other agents. Asking questions improves speed and success probability.
