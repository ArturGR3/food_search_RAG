# Recipe Recommendation System

This is a recipe recommendation system that suggests dishes based on user queries. It uses a combination of natural language processing and similarity search to find recipes that match the user's input.

## How to Use

1. Enter a query describing the kind of recipe you're looking for. For example:
   - "vegetarian pasta dish with tomatoes"
   - "spicy chicken curry"
   - "chocolate dessert for two"

2. The system will return the top 3 recipe recommendations, including:
   - An image of the dish
   - The recipe title
   - A list of ingredients
   - Cooking instructions

## Technology Stack

- Python
- Gradio (for the user interface)
- Sentence Transformers (for text embeddings)
- FAISS (for similarity search)
- Pandas (for data manipulation)

## Setup

To run this project locally, follow these steps:

1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Data

The recipe data and embeddings are stored in `data/recipe_embeddings.pkl`.
The FAISS index for similarity search is stored in `recipe_index.faiss`.
Images for the recipes are stored in the `Food Images` directory.

## License

This project is licensed under the MIT License.