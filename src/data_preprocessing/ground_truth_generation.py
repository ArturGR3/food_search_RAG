"""
Description: This script generates short, user-like prompts for recipe searches using OpenAI's Language Model API.

It uses asyncio for concurrent API calls and includes error handling and rate limiting.
"""

import sys
from pathlib import Path
import asyncio
import time
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
import argparse

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from src.utils.settings import get_settings
from src.utils.llm_factory import AsyncLLMFactory

# Initialize settings and LLM client
settings = get_settings()
async_llm_client = AsyncLLMFactory("openai")

# Print LLM client configuration
print(f"{async_llm_client.provider} client initialized with settings:")
print(f"Default model: {async_llm_client.settings.default_model}")
print(f"Temperature: {async_llm_client.settings.temperature}")
print(f"Max retries: {async_llm_client.settings.max_retries}")    

class RecipePrompt(BaseModel):
    """A schema for generating recipe prompts."""
    Index: int = Field(..., description="The index of the recipe")
    Title: str = Field(..., description="The title of the recipe")
    prompts: List[str] = Field(..., description="A list of 5 short, user-like prompts (up to 10 words each) for searching a recipe")

# Define system and user messages for the LLM
SYSTEM_MESSAGE = """
You are an AI assistant that generates short, user-like prompts for recipe searches.
Each prompt should be up to 5 words long, reflecting how a user might search for this recipe. 
Focus on key ingredients, cooking methods, or dish types. Make the prompts diverse and natural-sounding.
""".strip()

USER_MESSAGE = """
Generate 5 short, user-like prompts (up to 10 words each) 
that someone might use to search for a recipe with the following information:
""".strip()

async def generate_prompts(recipe_text: str) -> RecipePrompt:
    """
    Generate short, user-like prompts for a given recipe using OpenAI.

    Args:
        recipe_text (str): The recipe information to generate prompts for.

    Returns:
        RecipePrompt: A RecipePrompt object containing the generated prompts.
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = await async_llm_client.create_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": f"{USER_MESSAGE} {recipe_text}"}
                ],
                response_model=RecipePrompt,
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error generating prompts: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            else:
                print(f"Failed to generate prompts after {max_retries} attempts: {e}")
                return None
    
async def generate_and_save_prompts(semaphore: int, data_path: str, output_csv_path: str, num_recipes: int = 1000):
    """
    Generate and save prompts for a given number of recipes.

    Args:
        semaphore (int): The number of concurrent API calls allowed.
        data_path (str): Path to the input CSV file containing recipe data.
        output_csv_path (str): Path to save the output CSV file with generated prompts.
        num_recipes (int, optional): Number of recipes to process. Defaults to 1000.
    """
    sem = asyncio.Semaphore(semaphore)

    async def rate_limited_generate_prompts(text: str) -> RecipePrompt:
        async with sem:
            return await generate_prompts(text)
    
    df = pd.read_csv(data_path)
    selected_df = df.sample(n=num_recipes, random_state=42)
    
    tasks = [rate_limited_generate_prompts(f'text_for_search: {row["text_for_search"]}, Index: {row["Index"]}, Title: {row["Title"]}') for _, row in selected_df.iterrows()]
    
    results = await asyncio.gather(*tasks)
    prompts = [result.model_dump() for result in results if result]

    df_prompts = pd.DataFrame(prompts)
    df_prompts['Index'] = df_prompts['Index'].astype(df['Index'].dtype)
    df_prompts.to_csv(output_csv_path, index=False)

    print(f"Generated prompts for {len(df_prompts)} recipes and saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recipe prompts")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to process (default: 10, use 1000 for full dataset)")
    args = parser.parse_args()

    semaphore = 4
    
    is_full_dataset = args.samples == 1000
    output_filename = "prompts_dataframe.csv" if is_full_dataset else "prompts_dataframe_test.csv"
    
    start_time = time.time()
    asyncio.run(generate_and_save_prompts(
        semaphore,
        f'{project_root}/data/recipes.csv', 
        f'{project_root}/data/{output_filename}',
        num_recipes=args.samples
    ))
    end_time = time.time()
    print(f"Semaphore {semaphore}: Time taken: {end_time - start_time} seconds")

