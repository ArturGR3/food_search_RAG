# Description: This script preprocesses the data and generates embeddings for the recipes in the dataset.

import zipfile
import pickle
import pandas as pd 
import numpy as np
from tqdm import tqdm
import os 
import ast 
from sentence_transformers import SentenceTransformer
import faiss

# Getting data from Kaggle 
comp_name = 'food-ingredients-and-recipe-dataset-with-images'
data_path = '../data'

if os.path.exists('./data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'):
    print('Data already exists. Loading data...')
    df = pd.read_csv('./data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
else :
    print('Data does not exist. Unzipping data...')
    zip_file = os.path.join(data_path, f"{comp_name}.zip")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_path)
    df = pd.read_csv('./data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
df = pd.read_csv('./data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Combine the title, ingredients, and instructions into a single text for embedding
df['text_for_embedding'] = df.apply(lambda row: f"{row['Title']}. Ingredients: {', '.join(ast.literal_eval(row['Cleaned_Ingredients']))}. Instructions: {row['Instructions']}", axis=1)

# Save the preprocessed data
# df.to_csv('../data/preprocessed_recipes.csv', index=False)

# Load the preprocessed data
# df = pd.read_csv('../data/preprocessed_recipes.csv')

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings in batches (potentially rewrite it as a parallelized function)
def generate_embeddings(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Generate embeddings
print("Generating embeddings...")
embeddings = generate_embeddings(df['text_for_embedding'].tolist())

# Add embeddings to the dataframe
df['embedding'] = embeddings.tolist()

# Convert embeddings to numpy array if they're not already
embeddings = np.array(embeddings).astype('float32')

# Create the FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
print("Saving FAISS index...")
faiss.write_index(index, '../data/recipe_index.faiss')

# Save the dataframe with embeddings to a pickle file if it does not exist
if not os.path.exists('./data/recipe_embeddings.pkl'):
    print("Saving embeddings...")
    with open('../data/recipe_embeddings.pkl', 'wb') as f:
        pickle.dump({'df': df, 'embeddings': embeddings}, f)
else:
    # Load the existing embeddings
    print("Loading embeddings...")
    with open('./data/recipe_embeddings.pkl', 'rb') as f:
        recipe = pickle.load(f)
    
embeddings = recipe['embeddings']
df = recipe['df']
print("\nEmbedding shape:", embeddings.shape)
print("Dataframe shape:", df.shape)


