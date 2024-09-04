import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

def load_data_and_index(embeddings_path, index_path):
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
        df = data['df']
        embeddings = data['embeddings']
    
    index = faiss.read_index(index_path)
    
    return df, embeddings, index

def recommend_recipes(query, df, embeddings, index, model, top_k=3):
    query_vector = model.encode([query])[0].astype('float32')
    query_vector = np.array([query_vector]).astype('float32')
    
    D, I = index.search(query_vector, top_k)
    
    recommendations = df.iloc[I[0]]
    
    results = []
    for _, recipe in recommendations.iterrows():
        image_path = os.path.join('Food Images', recipe['Image_Name'] + '.jpg')
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
    embeddings_path = 'data/recipe_embeddings.pkl'
    index_path = 'recipe_index.faiss'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df, embeddings, index = load_data_and_index(embeddings_path, index_path)

def get_recommendations(query):
    global df, embeddings, index, model
    if df is None or embeddings is None or index is None or model is None:
        initialize()
    
    recommendations = recommend_recipes(query, df, embeddings, index, model)
    return recommendations