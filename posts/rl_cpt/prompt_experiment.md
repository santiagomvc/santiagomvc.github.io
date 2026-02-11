# CPT vs EV Experiment Execution Plan

This document is the operating manual for a team of agents executing experiments that compare rational (Expected Value / REINFORCE) and descriptive (Cumulative Prospect Theory / CPT-PG) decision-making in a custom Cliff Walking environment.

---

## Project Goal

Find measurable behavioral differences between a rational REINFORCE agent and a human-like CPT-PG agent. The primary signal is **which row (path) the agent chooses to traverse** — some rows are riskier (closer to cliff), some rows are safer.

---

## Team Structure

```
Lead Agent (resource coordinator, main orchestrator)
├── exp1-agent: Risk Aversion for High-Probability Gains
├── exp2-agent: Risk Seeking for High-Probability Losses
├── exp3-agent: Risk Seeking for Low-Probability Gains
├── exp4-agent: Risk Aversion for Low-Probability Losses
└── exp5-agent: Loss Aversion (mixed domain)
```

Optional (stretch goals, only if experiments 1-5 succeed):
- exp6-agent: Allais Paradox Analog
- exp7-agent: Status Quo Bias
- exp8-agent: Endowment Effect

---

## Lead Agent Responsibilities

1. **Before experiments start**: Verify codebase changes are applied (deep merge in `utils.py`, output directory naming in `main.py` — these should already be done).
2. **Resource coordination**: Maximum **2 concurrent training runs**. More will exhaust memory.
3. **Execution waves**:
   - Wave 1: exp1-agent + exp2-agent (parallel)
   - Wave 2: exp3-agent + exp4-agent (parallel)
   - Wave 3: exp5-agent (solo)
   - Wave 4 (optional): exp6-8 agents
4. **Analytical search can always run**: Config search scripts use minimal resources and can run alongside training.
5. **Result validation**: After each experiment completes, review the path analysis output. A meaningful result shows >0.5 row difference in mean traversal row between agents, or clearly different path distributions.
6. **Final report**: Aggregate all experiment results into a summary.

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

## Per-Agent Workflow

Each experiment agent follows this workflow. **These starting configs are your initial setup — you WILL need to explore and adjust parameters.**

### Phase 1: Analytical Config Search (low resource, always allowed)

Adapt `scripts/find_divergent_config.py` to your experiment. Use `path_likelihood.py` functions:
- `cliff_fall_probability(row, nrows, ncols, wind_prob)` — P(cliff) for a row
- `calculate_path_cpt_value(outcomes, value_func, weighting_func)` — CPT value with proper decision weights
- `calculate_path_expected_value(outcomes)` — EV value
- `build_path_outcome_distributions(env_config)` — outcome distributions for each row
- `compare_value_frameworks(env_config, cpt_params)` — compare EV vs CPT preferences

Search the parameter space listed in your experiment section. Find configs where EV and CPT **diverge in the predicted direction**.

### Phase 2: Quick Training (2 seeds, resource-constrained)

1. Create config YAML in `configs/` directory
2. Run: `python main.py -c your_config_name`
3. Check stdout for path analysis
4. Check `outputs/` for training curves and eval GIFs

### Phase 3: Iterate (if needed)

If no behavioral difference:
- Adjust parameters from the exploration list
- Re-run analytical search with wider ranges
- Check training curves — did the agents converge?
- Try different grid sizes or wind probabilities
- Document what you tried and why

### Phase 4: Confirm (4 seeds)

Once a good config is found:
1. Update config: `n_seeds: 4`
2. Re-run for statistical confirmation
3. Pool all eval episodes across seeds

### Phase 5: Report

Document:
1. Final config (YAML)
2. Path distribution comparison (table)
3. All metrics
4. Hypothesis confirmed/rejected + reasoning
5. Any surprises or insights

---

## Experiment 1: Risk Aversion for High-Probability Gains (REQUIRED)

**Agent**: exp1-agent
**Wave**: 1 (parallel with Exp 2)

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

**Agent**: exp2-agent
**Wave**: 1 (parallel with Exp 1)

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

**Agent**: exp3-agent
**Wave**: 2 (parallel with Exp 4)

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

**Agent**: exp4-agent
**Wave**: 2 (parallel with Exp 3)

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

**Agent**: exp5-agent
**Wave**: 3 (solo)

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

**Agent**: exp6-agent
**Wave**: 4

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

**Agent**: exp7-agent
**Wave**: 4

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

**Agent**: exp8-agent
**Wave**: 4

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
* Large batch sizes is pretty important since some important estimators depend on the batch size. Don't have it lower than 64.
* After the experiment is done, save a single gift with the expected behavior for each agent.
* Prepare a good strategy for learning when we need high wind probabilities for the experiment, which will make learning harder (example: taller grids, different reward values, etc.)

## Important Reminders

1. **`stochasticity: windy`** — ALWAYS set this in configs that use wind_prob. Without it, wind is disabled.
2. **Positive domain requires `reward_step: 0`** — non-zero step rewards in positive domain cause unwanted behavior. Use gamma as the path-length penalty.
3. **Deep merge is active** — experiment configs only need to specify parameters that differ from base.yaml.
4. **2 concurrent training runs maximum** — inform the lead agent before starting training. Other agents can work in parallel on other tasks like documentation, calculations, proposing new experiments, etc.
5. **Iterate on configs** — the starting configs are educated guesses. You will likely need to adjust parameters. Run analytical search first, then quick training (2 seeds), then confirm (4 seeds).
6. **The main goal is behavioral differences** — a "successful" experiment shows CPT and REINFORCE choosing different paths, in the direction predicted by CPT theory.
7. **Document everything** — record what configs you tried, what worked, what didn't, and why.
8. The lead agent should **keep an eye on compute resources** and make sure we are not crashing our compute, controlling runs accordingly
9. **Multiple experiments and runs are expected** to reach a succesful config. Think deeply about the proposed experiments and the possible consequences before running.
10. Feel free to **read the research, review the codebase or run calculations as needed**.
10. Any **code changes that can break the experiments flow must be coordinated and confirmed with the lead agent** to avoid catastrophic changes.
11. **Feel free to ask any questions** you need to clarify or improve experimentation performance. This goes to the lead and all the other agents. Asking questions improves speed and success probability.
