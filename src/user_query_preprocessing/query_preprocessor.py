from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import sys

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from src.reporting.db import store_query_adjustment
from src.utils.llm_factory import LLMFactory

class PreprocessedQuery(BaseModel):
    is_recipe_request: bool = Field(..., description="Whether the query is a valid recipe request")
    original_query: str = Field(..., description="The original user query")
    adjusted_query: str = Field(..., description="The adjusted query with corrections and improvements")
    excluded_ingredients: List[str] = Field(default_factory=list, description="List of ingredients to be excluded")
    adjustments: List[str] = Field(default_factory=list, description="List of adjustments made to the query")
    reasoning: str = Field(..., description="Explanation of the preprocessing decisions")

def preprocess_query(query: str, llm_factory: LLMFactory, session_id: str) -> PreprocessedQuery:
    system_message = """
    You are an AI assistant specialized in preprocessing recipe queries. Your task is to:
    1. Determine if the query is a valid recipe request.
    2. Correct any typos or misspellings.
    3. Adjust the query to better suit a recipe search system.
    4. Identify and list any ingredients that should be excluded.
    5. List any adjustments made to the query.
    6. Provide reasoning for your decisions.

    Rules:
    - If the query is not related to recipes or food, mark it as not a recipe request.
    - For typos, suggest corrections but maintain the original intent of the query.
    - Expand abbreviations and clarify ambiguous terms.
    - Pay special attention to negations (e.g., "without", "no", "except").
    - For negations, remove the excluded ingredients from the adjusted query and add them to the excluded_ingredients list.
    - The adjusted query should focus on what the user wants, not what they don't want.
    - In the reasoning field, explain your thought process, especially for handling negations and any significant adjustments.

    Remember, do not use hardcoded rules for specific ingredients or dishes. Your approach should be general and applicable to any recipe query.
    """
    
    user_message = f"Preprocess the following query: {query}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    preprocessed_query = llm_factory.create_completion(
        response_model=PreprocessedQuery,
        messages=messages,
        stream=False
    )
    
    # Store the original and adjusted queries for review
    print(f"Storing query adjustment for session: {session_id}")
    store_query_adjustment(session_id, preprocessed_query.original_query, preprocessed_query.adjusted_query, preprocessed_query.excluded_ingredients)
    
    return preprocessed_query

# Example usage (for testing purposes)
if __name__ == "__main__":
    
    llm_factory = LLMFactory("groq")  # Or whichever provider you prefer
    
    test_queries = [
        "vegetarian pasta recipie",
        # "how to make a choclate cake",
        # "what's the weather like today",
        # "spicy chiken curry without nuts",
        # "recipe for gluten-free pizzza",
        # "salad with no chicken",
        # "soup without carrots or celery",
        # "dessert recipe excluding dairy",
    ]
    
    for query in test_queries:
        result = preprocess_query(query, llm_factory, "test_session")
        print(f"Original: {result.original_query}")
        print(f"Adjusted: {result.adjusted_query}")
        print(f"Is recipe request: {result.is_recipe_request}")
        print(f"Excluded ingredients: {result.excluded_ingredients}")
        print(f"Adjustments: {result.adjustments}")
        print(f"Reasoning: {result.reasoning}")
        print("---")