
import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import os
from src.utils.llm_factory import LLMFactory
from src.utils.settings import get_settings
from pydantic import BaseModel, Field

def load_data_and_index(embeddings_path, index_path):
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
        df = data['df']
        embeddings = data['embeddings']
    
    index = faiss.read_index(index_path)
    
    return df, embeddings, index

def recommend_recipes(query, df, index, model, top_k=3):
    """
    query: user query to find similar recipes
    df: dataframe containing the recipe data and embeddings
    index: FAISS index for similarity search
    model: SentenceTransformer model for encoding queries ('all-MiniLM-L6-v2')
    top_k: number of similar recipes to return
    """
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.array([query_vector]).astype('float32')
    
    D, I = index.search(query_vector, top_k)
    
    recommendations = df.iloc[I[0]]
    
    results = []
    for _, recipe in recommendations.iterrows():
        image_path = os.path.join('../data/Food Images', recipe['Image_Name'] + '.jpg')
        results.append({
            'Title': recipe['Title'],
            'Ingredients': recipe['Ingredients'],
            'Instructions': recipe['Instructions'],
            'Image_Path': image_path
        })
    
    return results

# Global variables to store loaded data
df, embeddings, index = None, None, None
model = None

def initialize():
    global df, embeddings, index, model 
    embeddings_path = '../data/recipe_embeddings.pkl'
    index_path = '../data/recipe_index.faiss'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df, embeddings, index = load_data_and_index(embeddings_path, index_path)
    

def get_recommendations(query, top_k):
    global df, embeddings, index, model
    if df is None or embeddings is None or index is None or model is None:
        initialize()
    
    recommendations = recommend_recipes(query, df, index, model, top_k=top_k)
    return recommendations

settings = get_settings()
llm_client = LLMFactory("openai")

# show properties of llm_client
print(f"{llm_client.provider} client initialized with settings:")
print(f"Default model: {llm_client.settings.default_model}")
print(f"Temperature: {llm_client.settings.temperature}")
print(f"Max retries: {llm_client.settings.max_retries}")    

class IngredientCategories(BaseModel):
    spices: List[str] = Field(default_factory=list, description="List all spices, herbs, and seasonings")
    vegetables: List[str] = Field(default_factory=list, description="List of vegetables and fruits")
    proteins: List[str] = Field(default_factory=list, description="List of proteins (meats, tofu, beans, etc.)")
    other: List[str] = Field(default_factory=list, description="Other ingredients used in the recipe")

class BestRecipe(BaseModel):
    """A schema for generating recipe prompts with categorized ingredients."""
    title: str = Field(..., description="The title of the recipe")
    ingredients: IngredientCategories = Field(..., description="Categorized list of ingredients for the recipe")
    instructions: str = Field(..., description="The instructions for the recipe")
    equipment: List[str] = Field(default_factory=list, description="List of equipment needed for the recipe")
    # adjustments: Optional[str] = Field(None, description="Any adjustments for the recipe")

system_message = """
You are an expert chef who can pick the most relevent food recipes from the options provided. 
""".strip()

def format_user_content(query: str) -> str:
    recommendations = get_recommendations(query, top_k=5)
    
    recipes_str = "\n\n".join([
        f"Recipe {i+1}:\nTitle: {r['Title']}\nIngredients: {r['Ingredients']}\nInstructions: {r['Instructions']}"
        for i, r in enumerate(recommendations)
    ])
    print(recipes_str)
    
    return f"""As an experienced chef, analyze the following recipes for '{query}' and choose the best one:

{recipes_str}

Based on these recipes, provide the best recipe in the following format:
- Title: The title of the best recipe (if the recipe is modified to better fit user need, mention it here)
- Ingredients: Categorize the ingredients into the following groups (ingredients can belong to one group):
  - Spices: List all spices, herbs, and seasonings
  - Vegetables: List all vegetables and fruits
  - Proteins: List all protein sources (meats, tofu, beans, etc.)
  - Other: List any other ingredients that don't fit into the above categories
** IF USER REQUESTED SPECIAL DIETARY NEEDS, MAKE SURE TO MARK INGREDIENTS WITH [Adjusted] **
- Equipment: List all equipment needed for the recipe
- Instructions: The step-by-step instructions for the recipe

Please ensure your response fits the updated BestRecipe model structure with categorized ingredients and separate equipment list.
"""


def augment_with_llm(query: str):
    
    user_content = format_user_content(query)
    
    response = llm_client.create_completion(
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        response_model=BestRecipe,
    )
    return response

get_recommendations('chicken',top_k=5)

respose = augment_with_llm("plant based chicken")
import json
print(json.dumps(respose.model_dump(), indent=2))