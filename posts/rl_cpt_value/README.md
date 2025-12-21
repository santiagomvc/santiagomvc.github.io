# Are AI agents aligned with human behavior - Part 1

I'm writing a blogpost about how the behavior of AI agents compares to human behavior. For that I'm going to run a experiment on three different RL environments using three different types of agents. This blogpost will focus on the value side of CPT, the next one will focus on the probability side, and finally we will combine both.

## Agents

* RL Agent (PPO): A simple RL agent using the PPO algorithm to learn a policy for the environment.
* Human Aligned Rl Agent (PPO): A RL agent which value functions are close to cumulative prospect theory.
* Large Language Model Agent (Opus-4.5): State-of-the-art Language Model with tool calling capabilities.

## Environments

* Cliff Walking: A simple environment where the agent has to navigate a grid of cliffs and rewards.
* Frozen Lake: A simple environment where the agent has to navigate a grid of lakes and rewards.
* Blackjack: A simple environment where the agent has to learn to play blackjack.

## Experiment

1. Build cliff walking env with random agent. It should include an option to visualize the environment and the agent's behavior.

2. Build an LLM based agent. The input should be the environment state and the output should be the action to take. The prompt should also include a description of the environment and the goal. Use Claude Opus 4.5 through the anthropic API.

3. Build a PPO agent. The input should be the environment state and the output should be the action to take. Build it using RLlib and be explicit about the configuration.

4. Write the functions to use for the CPT based agent. The functions user reward shaping to modify the rewards using the value function distortion defined in CPT.

5. Build a CPT-PPO based agent. The input should be the environment state and the output should be the action to take. Use the PPO agent as a baseline but the value function should be modified to use the CPT value function.

6. Train (if needed) the agents and run an evaluation with multiple episodes saving the results for cliff walking. Log training and evaluation results to Weights and Biases.

7. Build the frozen lake env.

8. Train (if needed) the agents and run an evaluation with multiple episodes saving the results for frozen lake. Log training and evaluation results to Weights and Biases.

9. Build the blackjack env.

10. Train (if needed) the agents and run an evaluation with multiple episodes saving the results for blackjack. Log training and evaluation results to Weights and Biases.

11. Build a table with the results for all the agents and environments.

12. Write a blogpost about the results.

## Future ideas

* RLVR on regular env and CPT distortions for a single env
    * Differnces with normal env RLVR behavior
    * Changes in behavior when acting on the other envs
    * Changes in LLM value completions
* RL CPT Probability distortion experiment
* Agent combining RL CPT probability and value changes.
* The "Trap" Corridor (Reference Points) env experiment.
