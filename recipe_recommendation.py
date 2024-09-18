import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import os
import instructor
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=True))

def load_data_and_index(embeddings_path, index_path):
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
        df = data['df']
        embeddings = data['embeddings']
    
    index = faiss.read_index(index_path)
    
    return df, embeddings, index

def format_ingredients(ingredients):
    ingredients = eval(ingredients)
    formatted = ""
    for i, ingredient in enumerate(ingredients, 1):
        formatted += f"{i}. {ingredient}\n"
    return formatted

def recommend_recipes(query, df, embeddings, index, model, top_k=3):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.array([query_vector]).astype('float32')
    
    D, I = index.search(query_vector, top_k) # Search for the top_k most similar recipes
    
    recommendations = df.iloc[I[0]]
    
    results = []
    for i, (_, recipe) in enumerate(recommendations.iterrows()):
        image_path = os.path.join('data/Food_Images', recipe['Image_Name'] + '.jpg')
        results.append({
            'Title': recipe['Title'],
            'Ingredients': format_ingredients(recipe['Ingredients']),
            'Instructions': recipe['Instructions'],
            'Image_Path': image_path,
            'cosine_similarity': 1 - float(D[0][i])  # Convert distance to similarity
        })
    
    # Sort results by cosine similarity in descending order
    results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    
    return results

# Global variables to store loaded data
df, embeddings, index = None, None, None
model = None

def initialize():
    global df, embeddings, index, model
    embeddings_path = 'data/recipe_embeddings.pkl'
    index_path = 'data/recipe_index.faiss'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df, embeddings, index = load_data_and_index(embeddings_path, index_path)

def get_recommendations(query):
    global df, embeddings, index, model
    if df is None or embeddings is None or index is None or model is None:
        initialize()
    
    recommendations = recommend_recipes(query, df, embeddings, index, model)
    return recommendations

class IngredientCategories(BaseModel):
    spices_herbs: List[str] = Field(default_factory=list, description="Spices, herbs and seasonings with portions")
    vegetables: List[str] = Field(default_factory=list, description="Vegetables and fruits with portions")
    meat_or_protein: List[str] = Field(default_factory=list, description="Meat and protein sources with portions")
    carbs : List[str] = Field(default_factory=list, description="Carbohydrates and grains with portions")
    other: List[str] = Field(default_factory=list, description="Other things like equipment or condiments")

client = instructor.from_groq(Groq(), mode=instructor.Mode.TOOLS)

def categorize_ingredients(ingredients):
    system_context = """
    You are an expert chef that can break down the ingredient list into categories.
    """
    
    categories = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        response_model=IngredientCategories,
        messages=[
            {"role": "system", "content": system_context},  
            {"role": "user", "content": f"Ingredient list: {ingredients}"},
        ],
    )
    return categories

class CookingInstructions(BaseModel):
    steps: List[str] = Field(default_factory=list)

def process_cooking_instructions(instructions: str) -> str:
    system_context = """
    You are an expert chef that can break down cooking instructions into clear, enumerated steps.
    Provide a list of concise, actionable steps.
    Do not include step numbers in the instructions themselves.
    """
    
    try:
        processed_instructions = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            response_model=CookingInstructions,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": f"Cooking instructions: {instructions}"},
            ],
        )
        
        # Format the steps into a numbered list
        formatted_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(processed_instructions.steps))
        return formatted_steps
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Unable to process instructions due to an error."

if __name__ == "__main__":
    # Example usage
    test = get_recommendations("Tofu soup")
    for i in test:
        print(f'Title:\n{i["Title"]}\n')
        print(f'Ingredients:\n{i["Ingredients"]}\n')
        print(f'Instructions:\n{i["Instructions"]}\n')
        print(f'Image Path:\n{i["Image_Path"]}\n')
        print(f'Cosine Similarity:\n{i["cosine_similarity"]}\n')
        print('\n')

    # Example of categorizing ingredients
    ingredients = test[0]["Ingredients"]
    categories = categorize_ingredients(ingredients)
    print("Ingredient Categories:")
    print(categories)

    # Example of processing cooking instructions
    instructions = test[0]["Instructions"]
    processed_instructions = process_cooking_instructions(instructions)
    print("\nProcessed Cooking Instructions:")
    print(processed_instructions)