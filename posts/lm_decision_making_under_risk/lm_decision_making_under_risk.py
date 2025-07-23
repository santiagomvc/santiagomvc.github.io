# Load libraries
import json
import pandas as pd
import openai
import anthropic
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
# Enable logging at the INFO level
logging.basicConfig(level=logging.INFO)
# Load environment variables
load_dotenv("../../.env")

# Instantiate OpenAI and Anthropic clients
openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()
# Instantiate small models
llama_3_2_1b_client = InferenceClient(model="meta-llama/Llama-3.2-1B", timeout=120)
llama_3_2_1b_instruct_client = InferenceClient(model="meta-llama/Llama-3.2-1B-instruct", timeout=120)

# Load data
df = pd.read_csv("choices_data.csv")

# Tool definition
# NOTE: Small no instruct model can not return tool calls
tools = [
    {
        "type": "function",
        "function": {
            "name": "print_selected_option",
            "description": "Print the selected options from the choices presented",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_option": {
                        "type": "string",
                        "enum": ["A", "B"],  # This restricts the options to only A and B
                        "description": "The selected option from the choices presented",
                    },
                },
                "required": ["selected_option"],
            },
        },
    }
]
claude_tools = [
    {
        "name": "print_selected_option",
        "description": "Print the selected options from the choices presented",
        "input_schema": {
            "type": "object",
            "properties": {
                "selected_option": {
                    "type": "string", 
                    "enum": ["A", "B"],  # This restricts the options to only A and B
                    "description": "The selected option from the choices presented"
                }
            },
            "required": ["selected_option"]
        }
    }
]

# Define prompts
prompts = {
    "simple_choice": (
        """
        You must choose between the following options:

        Option A: {option_a}
        Option B: {option_b}
        """
    ), 
    "human_like_choice": (
        """
        You must choose between the following options what you think a human would pick:

        Option A: {option_a}
        Option B: {option_b}
        """
    ),
    "optimal_choice": (
        """
        You must choose between the following options what you think is optimal:

        Option A: {option_a}
        Option B: {option_b}
        """
    ),
    "cot_choice": (
        """
        You must choose between the following options and explain your choice. Let's think step by step:

        Option A: {option_a}
        Option B: {option_b}
        """
    )
}

results = []
for prompt_name, prompt in prompts.items():
    print("#----------------------------------#")
    print(f"Prompt: {prompt_name}")
    print("#----------------------------------#")
    for index, choice_data in df.iterrows():
        # Format prompt
        prompt_with_data = prompt.format(**choice_data)

        # Small instruct model
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt_with_data,
            },
        ]
        response = llama_3_2_1b_instruct_client.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="print_selected_option",
            max_tokens=500,
        )
        response_content = response.choices[0].message.tool_calls[0].function.arguments["selected_option"]
        results.append({
            "question_id": choice_data["id"],
            "prompt_name": prompt_name,
            "model": "llama-3.2-1b",
            "response": response_content,
            "ev_choice": int(choice_data["ev_choice"] == response_content),
            "pt_choice": int(choice_data["pt_choice"] == response_content),
        })
        print(f"{prompt_name}_llama-3.2-1b: {response_content}")

        # OpenAI 4o
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_with_data}],
            tools=tools,
            tool_choice="required",
        )
        response_content = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["selected_option"]
        results.append({
            "question_id": choice_data["id"],
            "prompt_name": prompt_name,
            "model": "gpt-4o",
            "response": response_content,
            "ev_choice": int(choice_data["ev_choice"] == response_content),
            "pt_choice": int(choice_data["pt_choice"] == response_content),
        })
        print(f"{prompt_name}_gpt-4o: {response_content}")

        # Anthropic Claude 3.5 Sonnet
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt_with_data}],
            tools=claude_tools,
            tool_choice={"type": "tool", "name": "print_selected_option"},
        )
        response_content = response.content[0].input["selected_option"]
        results.append({
            "question_id": choice_data["id"],
            "prompt_name": prompt_name,
            "model": "sonnet-3.5",
            "response": response_content,
            "ev_choice": int(choice_data["ev_choice"] == response_content),
            "pt_choice": int(choice_data["pt_choice"] == response_content),
        })
        print(f"{prompt_name}_sonnet-3.5: {response_content}")

        # OpenAI o1-preview
        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt_with_data}],
            tools=tools,
            tool_choice="required",
        )
        response_content = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["selected_option"]
        results.append({
            "question_id": choice_data["id"],
            "prompt_name": prompt_name,
            "model": "o3-mini",
            "response": response_content,
            "ev_choice": int(choice_data["ev_choice"] == response_content),
            "pt_choice": int(choice_data["pt_choice"] == response_content),
        })
        print(f"{prompt_name}_o3-mini: {response_content}")

# Results as a dataframe
results_df = pd.DataFrame(results)

# Plot results
cols = ["ev_choice", "pt_choice"]
# Plot by prompt name
prompt_results = results_df.groupby("prompt_name")[cols].mean().reset_index()
prompt_results.plot(kind="bar", x="prompt_name", y=cols)
plt.show()
# Plot by model
model_results = results_df.groupby("model")[cols].mean().reset_index()
model_results.plot(kind="bar", x="model", y=cols)
plt.show()
# Plot by model and prompt name
model_prompt_results = results_df.groupby(["model", "prompt_name"])[cols].mean().reset_index()
model_prompt_results["model_prompt"] = model_prompt_results["model"] + " / " + model_prompt_results["prompt_name"]
model_prompt_results.plot(kind="bar", x="model_prompt", y=cols)
plt.show()
# Save data
results_df.to_csv("responses.csv", index=False)
