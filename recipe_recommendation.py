import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Generator, Tuple
from sentence_transformers import SentenceTransformer
import os
from pydantic import BaseModel, Field
from llm_factory import LLMFactory

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

class RecipeAssessment(BaseModel):
    title: str = Field(..., description="Title of the recipe")
    is_relevant: bool = Field(..., description="Whether the recipe is relevant to the query")
    reason: str = Field(..., description="Reason for the relevance decision")
    relevance_score: int = Field(..., ge=0, le=10, description="Relevance score from 0 to 10")

class RelevanceAssessmentResult(BaseModel):
    assessments: List[RecipeAssessment] = Field(..., description="List of recipe assessments")

def retrieve_recipes(query: str, df: pd.DataFrame, embeddings: np.ndarray, index: faiss.Index, model: SentenceTransformer, top_k: int = 10) -> List[Dict[str, Any]]:
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.array([query_vector]).astype('float32')
    
    D, I = index.search(query_vector, top_k)
    
    recommendations = df.iloc[I[0]]
    
    results = []
    for i, (_, recipe) in enumerate(recommendations.iterrows()):
        image_path = os.path.join('data/Food_Images', recipe['Image_Name'] + '.jpg')
        recipe_dict = {
            'Title': recipe['Title'],
            'Ingredients': format_ingredients(recipe['Ingredients']),
            'Instructions': recipe['Instructions'],
            'Image_Path': image_path,
            'cosine_similarity': 1 - float(D[0][i])
        }
        results.append(recipe_dict)
    
    return results

def assess_recipes(query: str, recipes: List[Dict[str, Any]], excluded_ingredients: List[str], llm_factory: LLMFactory) -> RelevanceAssessmentResult:
    system_message = """
    You are an AI assistant specialized in assessing the relevance of recipes to user queries. 
    Your task is to determine if given recipes are relevant to the user's query, provide a reason for your decision, and assign a relevance score.
    Consider factors such as:
    1. How well the recipe ingredients and instructions match the query
    2. Any dietary restrictions or preferences implied in the query
    3. The overall style or type of dish requested
    4. Excluded ingredients specified by the user

    For each recipe, provide:
    1. Whether it's relevant (true/false)
    2. A brief reason for your decision
    3. A relevance score from 0 to 10, where 10 is most relevant

    Sort the recipes by relevance, with the most relevant recipes first.
    """.strip()
    
    recipes_text = "\n\n".join([
        f"Recipe {i+1}:\nTitle: {recipe['Title']}\nIngredients: {recipe['Ingredients']}\nInstructions: {recipe['Instructions']}"
        for i, recipe in enumerate(recipes)
    ])
    
    user_message = f"""
    Query: {query}
    Excluded Ingredients: {', '.join(excluded_ingredients)}

    Recipes:
    {recipes_text}

    Assess the relevance of each recipe to the query, considering the excluded ingredients.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    assessment_result = llm_factory.create_completion(
        response_model=RelevanceAssessmentResult,
        messages=messages
    )
    
    return assessment_result

def combine_relevant_recipes(recipes: List[Dict[str, Any]], assessments: RelevanceAssessmentResult, n: int = 3) -> List[Dict[str, Any]]:
    # Merge assessment results with recipe data
    for recipe, assessment in zip(recipes, assessments.assessments):
        recipe['is_relevant'] = assessment.is_relevant
        recipe['relevance_reason'] = assessment.reason
        recipe['relevance_score'] = assessment.relevance_score
    
    # Sort recipes by relevance_score (descending) and filter relevant ones
    relevant_recipes = [r for r in recipes if r['is_relevant']]
    relevant_recipes.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return relevant_recipes[:n]  # Return up to n most relevant recipes

def recommend_recipes(query: str, excluded_ingredients: List[str], df: pd.DataFrame, embeddings: np.ndarray, index: faiss.Index, model: SentenceTransformer, llm_factory: LLMFactory, n: int = 3) -> List[Dict[str, Any]]:
    # Step 1: Retrieve recipes using FAISS
    retrieved_recipes = retrieve_recipes(query, df, embeddings, index, model)
    
    # Step 2: Assess recipes
    assessment_result = assess_recipes(query, retrieved_recipes, excluded_ingredients, llm_factory)
    
    # Step 3: Combine relevant recipes and return n recipes
    recommended_recipes = combine_relevant_recipes(retrieved_recipes, assessment_result, n)
    
    return recommended_recipes

# Global variables to store loaded data
df, embeddings, index = None, None, None
model = None

def initialize():
    global df, embeddings, index, model
    embeddings_path = 'data/recipe_embeddings.pkl'
    index_path = 'data/recipe_index.faiss'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df, embeddings, index = load_data_and_index(embeddings_path, index_path)

def get_recommendations(query: str, excluded_ingredients: List[str], llm_factory: LLMFactory, n: int = 3) -> Dict[str, Any]:
    global df, embeddings, index, model
    if df is None or embeddings is None or index is None or model is None:
        initialize()
    
    # Step 1: Retrieve recipes using FAISS
    retrieved_recipes = retrieve_recipes(query, df, embeddings, index, model, top_k=10)
    
    # Step 2: Assess recipes
    assessment_result = assess_recipes(query, retrieved_recipes, excluded_ingredients, llm_factory)
    
    # Step 3: Combine relevant recipes and return n recipes
    recommended_recipes = combine_relevant_recipes(retrieved_recipes, assessment_result, n)
    
    return {
        "retrieved_recipes": retrieved_recipes,
        "assessed_recipes": assessment_result.assessments,
        "recommended_recipes": recommended_recipes
    }
    
def get_recommendations_stream(query: str, excluded_ingredients: List[str], llm_factory: LLMFactory, n: int = 3) -> Generator[Tuple[str, Any], None, None]:
    global df, embeddings, index, model
    if df is None or embeddings is None or index is None or model is None:
        initialize()
    
    # Step 1: Retrieve recipes using FAISS
    retrieved_recipes = retrieve_recipes(query, df, embeddings, index, model, top_k=10)
    yield "retrieved_recipes", retrieved_recipes
    
    # Step 2: Assess recipes
    assessment_result = assess_recipes(query, retrieved_recipes, excluded_ingredients, llm_factory)
    yield "assessed_recipes", assessment_result.assessments
    
    # Step 3: Combine relevant recipes and return n recipes
    recommended_recipes = combine_relevant_recipes(retrieved_recipes, assessment_result, n)
    yield "recommended_recipes", recommended_recipes

if __name__ == "__main__":
    # Example usage
    from llm_factory import LLMFactory
    llm_factory = LLMFactory("groq")  # Or whichever provider you prefer
    
    test_query = "salad without tomatoes"
    test_excluded_ingredients = ["tomatoes"]
    
    # Step by step example
    recipes_retrieved = retrieve_recipes(test_query, df, embeddings, index, model, top_k=5)
    recipes_assesed = assess_recipes(test_query, recipes_retrieved, test_excluded_ingredients, llm_factory)
    for i, assessment in enumerate(recipes_assesed.assessments):
        print(f"Assessment {i}:")
        print(f"Title: {recipes_retrieved[i]['Title']}")
        print(f"Relevance: {assessment.is_relevant}")
        print(f"Reason: {assessment.reason}")
        print(f"Relevance Score: {assessment.relevance_score}")
        print("\n" + "="*50 + "\n")
    
    
    recommendations = get_recommendations(test_query, test_excluded_ingredients, llm_factory)
    
    for i, recipe in enumerate(recommendations):
        print(f"Recommendation {i}:")
        print(f"Title: {recipe['Title']}")
        print(f"Relevance Score: {recipe['relevance_score']}")
        print(f"Relevance Reason: {recipe['relevance_reason']}")
        print(f"Cosine Similarity: {recipe['cosine_similarity']}")
        print(f"Ingredients:\n{recipe['Ingredients']}")
        print(f"Instructions:\n{recipe['Instructions']}")
        print(f"Image Path: {recipe['Image_Path']}")
        print("\n" + "="*50 + "\n")