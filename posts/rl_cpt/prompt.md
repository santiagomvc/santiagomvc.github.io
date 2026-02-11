# Reinforcement learning and Cumulative Prospect Theory Experiment

We want to run some experiments evaluating the differences in normative behavior (modeled through expected value) and descriptive human behavior (modeled through Cumulative Prospect Theory, papers related: posts/rl_cpt/research/prospect_theory.pdf and posts/rl_cpt/research/cumulative_prospect_theory.pdf) learned with a Reinforcement Learning agent (using the REINFORCE algorithm). For this goal we have create two main agents: REINFORCEAgent, a traditional RL agent using the reinforce algorithm; and CPTPGAgent, a modified RL Reinforce agent that takes episode information and applies CPT Value transformation to returns and Probability distortions to an empirically learned distribution of outcomes (proposed in this paper posts/rl_cpt/research/cpt-pg.pdf). 

For this experiment we are using a modified version of the Cliff Walking Gym environment. The main modifications are:

* Falling into the cliff ends the episode with the falling cliff reward.
* There's a wind variable representing the random probability of instead of moving as the agent decided the agent moves one step downward. The probability can be zero, which disables the wind.
* We can change the reward schema so rewards are all negative, positive or mixed.
* We have included a goal reward variable so we can change what the agent receives when it successfully navigates the environment.
* In an all positive environment, the reward structure is incredibly important to avoid unwanted behavior. What we have found is that our step reward should be zero, our cliff reward should be lower than the goal reward, and we should use the discount factor to balance the tradeoff between safe and risky paths.

## Goal

Create a group of agents that analyze the objective, research, and  codebase and propose a set of experiments that we should run where we could see the difference in behavior between a pure rational EV agent and a more descriptive CPT agent. The experiments must be run in our custom Cliff Walking environment, and the variables you can change are:

# --- env ---
env:
  shape: [4, 12]
  reward_cliff: -100
  reward_step: -1
  reward_goal: -1
  wind_prob: 0.1

# --- training ---
training:
  timesteps: 350000
  n_eval_episodes: 4
  batch_size: 8
  entropy_coef: 0.5
  entropy_coef_final: 0.01
  n_seeds: 1

# --- agent defaults ---
agent_config:
  lr: 0.0001
  gamma: 0.99
  baseline_type: ema

The experiments should be ordered with two criteria in mind: How much do the experiment actually showcases a difference in behavior between rational agents and humans, and how complex it is to actually write and run the experiments given the constrains we have (current code, training limits of reinforce, etc.). Some of the possible experiments I have gathered based on my research are:

* The fourfold pattern of risk attitudes is perhaps the most distinctive prediction. It emerges from the interaction of the S-shaped value function and the inverse S-shaped weighting function: risk aversion for high-probability gains, risk seeking for high-probability losses, risk seeking for low-probability gains, and risk aversion for low-probability losses. 
* Loss aversion — Losses loom roughly twice as large as equivalent gains. The paper estimates a loss-aversion coefficient of about 2.25, meaning people need roughly $225 in potential gains to accept a 50/50 chance of losing $100.
* Nonlinear preferences — EU assumes utility is linear in probabilities. Allais's paradox showed that the difference between .99 and 1.00 probability has far more psychological impact than the difference between .10 and .11, even though both are .01 gaps.
* Status quo bias
* Endowment effect

Analyze how our current CPT-PG implementation approaches the Baseline variable of CPT. Do we need to add something in our configuration to play with that variable. All experiments should be controlled by a config, making them simpler to run, document and repeat. Split The fourfold pattern into 4 experiments and add them in the begining of the list.

## Output

The output is a detailed plan document where we specify:

* The specific experiments we will run, including the proposed config
* The expected outcomes of those experiments
* The specific changes we need to do in our codebase to run the experiments. We want to keep our codebase simple but functional. This is not going to production environments.
* A description of the group of agents required to accomplish this project (ex: one agent per experiment; a planner, research, validator, and developer agent, etc.). Ideally each agent can work independently on an experiment so we can run it in parallel
* Take memory consumption and resource constrains into the plan. running multiple threads in parallel can brick the machine. you can use the main agent as a resource coordinator, trying to optimize resources and speed.

The outcome of this document will be the input of a team of agents that will develop this project.
Create a team of agents (researched, developer, etc.) to create this plan. Feel free to ask any questions you need.
