import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel
from typing import List, Dict, Any, Generator, Tuple
from pydantic import BaseModel, Field
from src.utils.llm_factory import LLMFactory
import os

def load_data(data_path: str) -> pd.DataFrame:
    csv_path = os.path.join(data_path)
    if os.path.exists(csv_path):
        print('Loading data from recipes.csv...')
        return pd.read_csv(csv_path)
    else:
        print('Data does not exist. Please ensure recipes.csv is in the data directory.')
        return None

class Recipes(LanceModel):
    """LanceDB schema for recipe data."""
    Index: int
    Title: str
    Cleaned_Ingredients: str
    Instructions: str
    Image_Name: str
    text_for_search: str

def initialize_lancedb(df: pd.DataFrame) -> lancedb.table.Table:
    """
    Initialize LanceDB with recipe data.

    Args:
        df (pd.DataFrame): Recipe data.

    Returns:
        lancedb.table.Table: Initialized LanceDB table.
    """
    db = lancedb.connect("./lancedb")
    table = db.create_table("recipes", mode="overwrite", schema=Recipes)
    
    data = df.copy()
    data['text_for_embedding'] = data['Title'] + ' ' + data['Cleaned_Ingredients'] + ' ' + data['Instructions']
    table.add(data[['Index', 'Title', 'Cleaned_Ingredients', 'Instructions', 'Image_Name', 'text_for_search']])
    table.create_fts_index("text_for_search", use_tantivy=False)
    
    return table

def format_ingredients(ingredients: str) -> str:
    """
    Format ingredients list for display. Each ingredient is numbered and formatted with a bullet point.

    Args:
        ingredients (str): String representation of ingredients list.

    Returns:
        str: Formatted ingredients list.
    """
    ingredients = eval(ingredients)
    return "\n".join(f"{i+1}. {ingredient}" for i, ingredient in enumerate(ingredients))

def retrieve_recipes(query: str, table: lancedb.table.Table, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recipes from LanceDB based on a search query.

    Args:
        query (str): Search query.
        table (lancedb.table.Table): LanceDB table.
        top_k (int): Number of recipes to retrieve.

    Returns:
        List[Dict[str, Any]]: List of retrieved recipes.
    """
    search_results = table.search(query, query_type='fts') \
        .select(["Index", "Title", "Cleaned_Ingredients", "Instructions", "Image_Name"]) \
        .limit(top_k) \
        .to_list()
    
    results = []
    for recipe in search_results:
        image_path = os.path.join('data/Food_Images', recipe['Image_Name'] + '.jpg')
        recipe_dict = {
            'Title': recipe['Title'],
            'Cleaned_Ingredients': format_ingredients(recipe['Cleaned_Ingredients']),
            'Instructions': recipe['Instructions'],
            'Image_Path': image_path,
        }
        results.append(recipe_dict)
    
    return results

class RecipeAssessment(BaseModel):
    """Model for recipe assessment results."""
    title: str = Field(..., description="Title of the recipe")
    is_relevant: bool = Field(..., description="Whether the recipe is relevant to the query")
    reason: str = Field(..., description="Reason for the relevance decision")
    relevance_score: int = Field(..., ge=0, le=10, description="Relevance score from 0 to 10")

class RelevanceAssessmentResult(BaseModel):
    """Model for overall relevance assessment results."""
    assessments: List[RecipeAssessment] = Field(..., description="List of recipe assessments")

def assess_recipes(query: str, recipes: List[Dict[str, Any]], excluded_ingredients: List[str], llm_factory: LLMFactory) -> RelevanceAssessmentResult:
    """
    Assess the relevance of retrieved recipes.

    Args:
        query (str): User's query.
        recipes (List[Dict[str, Any]]): Retrieved recipes.
        excluded_ingredients (List[str]): Ingredients to exclude.
        llm_factory (LLMFactory): LLM factory for creating completions.

    Returns:
        RelevanceAssessmentResult: Assessment results for recipes.
    """
    system_message = """
    You are an AI assistant specialized in assessing the relevance of recipes to user queries. 
    Your task is to determine if given recipes are relevant to the user's query, provide a reason for your decision, and assign a relevance score.
    Consider factors such as:
    1. How well the recipe ingredients and instructions match the query
    2. Any dietary restrictions or preferences implied in the query

    When considering excluded ingredients:
    - Compare them to the cleaned ingredients list of each recipe
    - Lower the relevance score if an excluded ingredient appears as part of a larger ingredient name (e.g., if "tomato" is excluded, lower the score for recipes containing "tomato paste", "cherry tomatoes", etc.)
    - Consider both exact matches and ingredients that contain the excluded item

    For each recipe, provide:
    1. Whether it's relevant (true/false)
    2. A brief reason for your decision
    3. A relevance score from 0 to 10, where 10 is most relevant

    Sort the recipes by relevance, with the most relevant recipes first.
    """.strip()
    
    recipes_text = "\n\n".join([
        f"Recipe {i+1}:\nTitle: {recipe['Title']}\nIngredients: {recipe['Cleaned_Ingredients']}\nInstructions: {recipe['Instructions']}"
        for i, recipe in enumerate(recipes)
    ])
    
    user_message = f"""
    Query: {query}
    Excluded Ingredients: {', '.join(excluded_ingredients)}

    Recipes:
    {recipes_text}

    Assess the relevance of each recipe to the query, considering the excluded ingredients as specified in the instructions. Be sure to lower the relevance score for recipes that contain any form of the excluded ingredients.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return llm_factory.create_completion(
        response_model=RelevanceAssessmentResult,
        messages=messages
    )

def combine_relevant_recipes(recipes: List[Dict[str, Any]], assessments: RelevanceAssessmentResult, max_recommendations: int = 3, min_relevance_score: int = 7) -> List[Dict[str, Any]]:
    """
    Combine and filter relevant recipes based on assessments.

    Args:
        recipes (List[Dict[str, Any]]): Retrieved recipes.
        assessments (RelevanceAssessmentResult): Assessment results.
        max_recommendations (int): Maximum number of recommendations to return.
        min_relevance_score (int): Minimum relevance score for a recipe to be considered.

    Returns:
        List[Dict[str, Any]]: List of relevant recipes.
    """
    for recipe, assessment in zip(recipes, assessments.assessments):
        recipe['is_relevant'] = assessment.is_relevant
        recipe['relevance_reason'] = assessment.reason
        recipe['relevance_score'] = assessment.relevance_score
    
    relevant_recipes = [r for r in recipes if r['is_relevant'] and r['relevance_score'] >= min_relevance_score]
    relevant_recipes.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return relevant_recipes[:max_recommendations]

# Global variables to store loaded data
df = None
lancedb_table = None

def initialize():
    """Initialize global variables for data and LanceDB table."""
    global df, lancedb_table
    df = load_data('data/recipes.csv')
    if df is not None:
        lancedb_table = initialize_lancedb(df)
    else:
        raise FileNotFoundError("recipes.csv not found in the data directory")

def get_recommendations(query: str, excluded_ingredients: List[str], llm_factory: LLMFactory, max_recommendations: int = 3, min_relevance_score: int = 7) -> Dict[str, Any]:
    """
    Get recipe recommendations based on a query.

    Args:
        query (str): User's query.
        excluded_ingredients (List[str]): Ingredients to exclude.
        llm_factory (LLMFactory): LLM factory for creating completions.
        max_recommendations (int): Maximum number of recommendations to return.
        min_relevance_score (int): Minimum relevance score for a recipe to be considered.

    Returns:
        Dict[str, Any]: Dictionary containing retrieved, assessed, and recommended recipes.
    """
    global df, lancedb_table
    if df is None or lancedb_table is None:
        initialize()
    
    retrieved_recipes = retrieve_recipes(query, lancedb_table)
    assessment_result = assess_recipes(query, retrieved_recipes, excluded_ingredients, llm_factory)
    recommended_recipes = combine_relevant_recipes(retrieved_recipes, assessment_result, max_recommendations, min_relevance_score)
    
    return {
        "retrieved_recipes": retrieved_recipes,
        "assessed_recipes": assessment_result.assessments,
        "recommended_recipes": recommended_recipes
    }


if __name__ == "__main__":
    # Example usage
    llm_factory = LLMFactory("groq")
    
    test_query = "soup"
    test_excluded_ingredients = ["onions"]
    
    recommendations = get_recommendations(test_query, test_excluded_ingredients, llm_factory)
    
    for i, recipe in enumerate(recommendations["recommended_recipes"]):
        print(f"Recommendation {i+1}:")
        print(f"\nTitle: {recipe['Title']}")
        print(f"\nRelevance Score: {recipe['relevance_score']}")
        print(f"\nRelevance Reason: {recipe['relevance_reason']}")
        print(f"\nIngredients:\n{recipe['Cleaned_Ingredients']}")
        print(f"\nInstructions:\n{recipe['Instructions']}")
        print(f"\nImage Path: {recipe['Image_Path']}")
        print("\n" + "="*50 + "\n")