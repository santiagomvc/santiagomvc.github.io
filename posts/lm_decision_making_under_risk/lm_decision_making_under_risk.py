# Load libraries
import pandas as pd
import openai
import anthropic
import torch
from transformers import pipeline
from dotenv import load_dotenv
import logging

# Enable logging at the INFO level
logging.basicConfig(level=logging.INFO)
# Load environment variables
load_dotenv("../../.env")

# Instantiate OpenAI and Anthropic clients
openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()

# Instantiate small models
llama_3_2_1b = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

llama_3_2_1b_instruct = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B-instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)


# Load data
df = pd.read_csv("choices_data.csv").head(1)

# Define prompt
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
    "explain_choice": (
        """
        You must choose between the following options and explain your choice. Let's think step by step:

        Option A: {option_a}
        Option B: {option_b}
        """
    )
}

for prompt_name, prompt in prompts.items():
    print(f"Prompt: {prompt_name}")
    for index, choice_data in df.iterrows():
        prompt_with_data = prompt.format(**choice_data)

        # Small no instruct model
        df.loc[index, f"{prompt_name}_raw_llama-3-2-1b"] = llama_3_2_1b(prompt_with_data, max_new_tokens=2000)[0]["generated_text"]
        print(f"{prompt_name}_raw_llama-3-2-1b\n", df.loc[index, f"{prompt_name}_raw_llama-3-2-1b"])
        print("#----------------------------------#")

        # Small instruct model
        df.loc[index, f"{prompt_name}_raw_llama-3-2-1b-instruct"] = llama_3_2_1b_instruct(prompt_with_data, max_new_tokens=2000)[0]["generated_text"]
        print(f"{prompt_name}_raw_llama-3-2-1b-instruct\n", df.loc[index, f"{prompt_name}_raw_llama-3-2-1b-instruct"])
        print("#----------------------------------#")

        # Get response from OpenAI 4o
        df.loc[index, f"{prompt_name}_raw_gpt-4o"] = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_with_data}],
        ).choices[0].message.content
        print(f"{prompt_name}_raw_gpt-4o\n", df.loc[index, f"{prompt_name}_raw_gpt-4o"])
        print("#----------------------------------#")

        # Get response from Anthropic
        df.loc[index, f"{prompt_name}_raw_claude-3-5-sonnet"] = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt_with_data}],
        ).content[0].text
        print(f"{prompt_name}_raw_claude-3-5-sonnet\n", df.loc[index, f"{prompt_name}_raw_claude-3-5-sonnet"])
        print("#----------------------------------#")

        # Get response from OpenAI o1-preview
        df.loc[index, f"{prompt_name}_raw_o1-preview"] = openai_client.chat.completions.create(
            model="o1-preview",
            messages=[{"role": "user", "content": prompt_with_data}],
        ).choices[0].message.content
        print(f"{prompt_name}_raw_o1-preview\n", df.loc[index, f"{prompt_name}_raw_o1-preview"])
        print("#----------------------------------#")

# Save data
df.to_csv("choices_data_with_responses.csv", index=False)
