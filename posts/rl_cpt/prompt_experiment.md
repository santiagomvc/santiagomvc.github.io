# CPT vs EV Experiment Execution Plan

This document is the operating manual for a team of agents executing experiments that compare rational (Expected Value / REINFORCE) and descriptive (Cumulative Prospect Theory / CPT-PG) decision-making in a custom Cliff Walking environment.

---

## Project Goal

Find measurable behavioral differences between a rational REINFORCE agent and a human-like CPT-PG agent. The primary signal is **which row (path) the agent chooses to traverse** — some rows are riskier (closer to cliff), some rows are safer.

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

1. **Analytical config search** (Phase 1): Adapt `scripts/find_divergent_config.py` for your assigned experiment. Search the parameter space to find configs where EV and CPT diverge in the predicted direction. This is lightweight and can always run.
2. **Config creation** (Phase 2): Create a YAML config in `configs/` based on your search results.
3. **Training runs** (Phase 3): Run `python main.py -c config_name`. **Always message the lead before starting a training run** so they can track the 2-concurrent-run limit.
4. **Initial assessment** (Phase 4): Check stdout path analysis, training curves, and eval GIFs. Determine if there is a meaningful behavioral difference (>0.5 row divergence or >15% path distribution shift).
5. **Iteration** (Phase 5): If no behavioral difference, adjust parameters and re-run. Document what you tried and why. If stuck after 3+ iterations, message the lead for guidance.
6. **Self-validation** (Phase 6): When you find a promising 2-seed result, validate it yourself — update the config to `n_seeds: 4`, re-run, pool eval episodes (4 seeds x 20 episodes = 80 datapoints), and compute statistical analysis (mean row, path distribution, Mann-Whitney U test). If validation fails (results are weak, inconsistent across seeds, or don't match hypothesis), **iterate immediately** — adjust parameters and go back to Phase 3. Do not wait for reassignment.
7. **Final report** (Phase 7): Once validation succeeds, produce a structured report (hypothesis confirmed/rejected/inconclusive, effect size, statistical significance) and message the lead with the final results.
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

---

## Metrics Framework

### Primary: Row Traversal Preference
Measured by `evaluate_paths()` in `utils.py`. For each eval episode, tracks the minimum row reached (the "path" chosen).

- **Mean traversal row**: Lower = safer. Higher = riskier.
- **Path distribution**: % of episodes at each row.
- **Row preference divergence**: Difference in mean traversal row between REINFORCE and CPT-PG.

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

Each experiment is owned end-to-end by a single researcher: **exploration** (Phases 1-4), **self-validation** (Phase 5-6), and **final report** (Phase 7). If validation fails, the researcher iterates immediately. The starting configs in each experiment section are initial guesses — researchers WILL need to explore and adjust parameters.

### Researcher Phases

#### Phase 1: Analytical Config Search (low resource, always allowed)

Adapt `scripts/find_divergent_config.py` to your experiment. Use `path_likelihood.py` functions:
- `cliff_fall_probability(row, nrows, ncols, wind_prob)` — P(cliff) for a row
- `calculate_path_cpt_value(outcomes, value_func, weighting_func)` — CPT value with proper decision weights
- `calculate_path_expected_value(outcomes)` — EV value
- `build_path_outcome_distributions(env_config)` — outcome distributions for each row
- `compare_value_frameworks(env_config, cpt_params)` — compare EV vs CPT preferences

Search the parameter space listed in your experiment section. Find configs where EV and CPT **diverge in the predicted direction**.

#### Phase 2: Config Creation

1. Create config YAML in `configs/` directory based on search results
2. Set `n_seeds: 2` for initial exploration

#### Phase 3: Quick Training (2 seeds, resource-constrained)

1. **Message the lead** to request a training slot
2. Run: `python main.py -c your_config_name`
3. Check stdout for path analysis
4. Check `outputs/` for training curves and eval GIFs
5. Assess: is there a meaningful behavioral difference? (>0.5 row divergence or >15% path distribution shift)

#### Phase 4: Iterate or Handoff

**If no behavioral difference** (iterate):
- Adjust parameters from the exploration list in your experiment section
- Re-run analytical search with wider ranges
- Check training curves — did the agents converge?
- Try different grid sizes or wind probabilities
- Document what you tried and why
- After 3+ failed iterations, message the lead for guidance

**If promising result found** (proceed to self-validation):
- Move directly to Phase 5 — do not hand off to another researcher.

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
Researcher                     Lead
    |                            |
    |-- Phase 1-2 (config) ---  |
    |-- "need train slot" ----> |
    |                            |-- "slot approved"
    |-- Phase 3 (2-seed run) -- |
    |-- Phase 4 (assess) ----  |
    |                            |
    |   [if weak: loop back to Phase 3, iterate params]
    |                            |
    |-- Phase 5 (4-seed run) -- |
    |                            |
    |   [if validation fails: loop back to Phase 3]
    |                            |
    |-- Phase 6 (stats) ------  |
    |-- "final report" -------> |
    |-- "request next exp" ---> |
    |                            |
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

### Config Search Approach

Adapt `scripts/find_divergent_config.py`: search for the gamma/goal_reward combination where EV prefers risky but CPT prefers safe. Use `path_likelihood.py` `calculate_path_cpt_value()` with a `CPTWeightingFunction` for proper probability weighting (the existing search script uses simplified CPT without probability weighting).

### Expected Outcome

- **REINFORCE**: Prefers risky path (row 2) — correctly computes higher EV despite small risk
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

### Expected Outcome

- **REINFORCE**: Prefers safe path (minimizes expected loss)
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

**This is the hardest experiment.** The probability overweighting effect is strong theoretically but may be hard to learn. If the risky path almost never succeeds during training, the agent may never explore it enough. Consider:
- Starting with moderate wind and gradually increasing
- Using very large goal rewards (1000+) to amplify the signal
- Using row 1 (d=2) instead of row 2 (d=1) for intermediate risk levels

### Expected Outcome

- **REINFORCE**: Strong safe-path preference (correctly penalizes low probability)
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

### Expected Outcome

- **REINFORCE**: May accept row 2 (tiny risk, shorter path, fewer step losses)
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

### Expected Outcome

- **REINFORCE**: Evaluates gamble symmetrically, may accept risky path
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

## Important Reminders

### Technical
1. **`stochasticity: windy`** — ALWAYS set this in configs that use wind_prob. Without it, wind is disabled.
2. **Positive domain requires `reward_step: 0`** — non-zero step rewards in positive domain cause unwanted behavior. Use gamma as the path-length penalty.
3. **Deep merge is active** — experiment configs only need to specify parameters that differ from base.yaml.

### Resource Management
4. **2 concurrent training runs maximum** — researchers MUST message the lead before starting any training run. Analytical config search scripts do NOT count (they are lightweight). The lead tracks active slots and approves/denies requests.
5. The lead should **actively monitor compute resources** and preemptively manage the training queue to prevent crashes.

### Experiment Execution
6. **Iterate on configs** — the starting configs are educated guesses. Researchers will likely need to adjust parameters. Follow the lifecycle: analytical search first, then quick training (2 seeds), then self-validate with confirmation (4 seeds). If validation fails, iterate immediately.
7. **The main goal is behavioral differences** — a "successful" experiment shows CPT and REINFORCE choosing different paths, in the direction predicted by CPT theory.
8. **Multiple experiments and runs are expected** to reach a successful config. Think deeply about the proposed experiments and the possible consequences before running.
9. **Document everything** — record what configs you tried, what worked, what didn't, and why.

### Team Coordination
10. **Researchers own experiments end-to-end** — each researcher explores, validates, and reports their own experiment. After completing a validated experiment, message the lead for your next assignment. Do not sit idle.
11. **Self-validation keeps momentum** — researchers validate their own results and iterate immediately on failure, avoiding handoff delays. The only resource constraint is the 2-concurrent-run limit.
12. **Any code changes that can break the experiments flow must be coordinated and confirmed with the lead** to avoid catastrophic changes.
13. **Cross-agent communication is encouraged** — share parameter insights, ask questions, and flag issues. Asking questions improves speed and success probability. Use direct messages for targeted info, not broadcasts.
14. Feel free to **read the research, review the codebase, or run calculations as needed**.

**Feel free to ask any questions** you need to clarify or improve experimentation performance. This goes to the lead and all the other agents. Asking questions improves speed and success probability.
