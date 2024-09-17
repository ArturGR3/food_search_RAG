# Description: This script generates short, user-like prompts for recipe searches using OpenAI's Language Model API.

import pandas as pd
import pickle
import json
import random
import time
from tqdm import tqdm
from settings import get_settings
from llm_factory import LLMFactory
from pydantic import BaseModel, Field
from typing import List, Dict
import os

settings = get_settings()
llm_client = LLMFactory("openai")

# show properties of llm_client
print(f"{llm_client.provider} client initialized with settings:")
print(f"Default model: {llm_client.settings.default_model}")
print(f"Temperature: {llm_client.settings.temperature}")
print(f"Max retries: {llm_client.settings.max_retries}")    

class RecipePrompt(BaseModel):
    """A schema for generating recipe prompts."""
    prompts: List[str] = Field(..., description="A list of 5 short, user-like prompts (up to 10 words each) for searching a recipe")

system_message = """
You are an AI assistant that generates short, user-like prompts for recipe searches.
Each prompt should be up to 5 words long, reflecting how a user might search for this recipe. 
Focus on key ingredients, cooking methods, or dish types. Make the prompts diverse and natural-sounding.""".strip()

user_message = """
Generate 5 short, user-like prompts (up to 10 words each) 
that someone might use to search for a recipe with the following information:""".strip()

def generate_prompts(recipe_text: str) -> List[str]:
    """Generate short, user-like prompts for a given recipe using OpenAI."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = llm_client.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"{user_message} {recipe_text}"
                    }
                ],
                response_model=RecipePrompt,
            )
            return response.prompts
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error generating prompts: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to generate prompts after {max_retries} attempts: {e}")
                return []

def generate_and_save_prompts(data_path: str, output_path: str, num_recipes: int = 1000):
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    df = data['df']

    # Randomly select recipes
    selected_indices = random.sample(range(len(df)), num_recipes)
    selected_df = df.iloc[selected_indices].reset_index(drop=True)

    # Load existing prompts if the file exists
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            all_prompts = json.load(f)
    else:
        all_prompts = {}

    # Generate prompts
    for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Generating prompts"):
        prompt_id = f"recipe_{idx}"
        if prompt_id not in all_prompts:
            prompts = generate_prompts(row['text_for_embedding'])
            if prompts:
                all_prompts[prompt_id] = {
                    'title': row['Title'],
                    'prompts': prompts
                }
                
                # Save prompts after each successful generation
                with open(output_path, 'w') as f:
                    json.dump(all_prompts, f, indent=2)
            else:
                print(f"Failed to generate prompts for recipe {idx}. Skipping...")
        else:
            print(f"Prompts for recipe {idx} already exist. Skipping...")

    print(f"Generated prompts for {len(all_prompts)} recipes and saved to {output_path}")

if __name__ == "__main__":
    generate_and_save_prompts('data/recipe_embeddings.pkl', 'data/generated_prompts.json', num_recipes=1000)
    
    # show 2 example prompts
    # with open('data/generated_prompts.json', 'r') as f:
    #     prompts = json.load(f)
    
    # for i, (recipe_id, data) in enumerate(prompts.items()):
    #     print(f"Recipe {i+1} - {data['title']}:")
    #     for j, prompt in enumerate(data['prompts']):
    #         print(f"Prompt {j+1}: {prompt}")
    #     print("\n")
    #     if i == 1:
    #         break
        
        
        