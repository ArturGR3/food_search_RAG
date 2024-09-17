import pandas as pd
import numpy as np
import faiss
import pickle
import json
import time
import random
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(file_path: str) -> pd.DataFrame:
    """Load the recipe data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['df']

# df = load_data('data/recipe_embeddings.pkl')

def load_prompts(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load the generated prompts from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
# prompts = load_prompts('data/generated_prompts.json')

def embed_recipes(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Embed recipes using SentenceTransformer."""
    return model.encode(df['text_for_embedding'].tolist(), show_progress_bar=True)

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a FAISS index from embeddings."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_recipes_faiss(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 5) -> List[int]:
    """Retrieve top-k similar recipes using FAISS."""
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return I[0].tolist()

def calculate_mrr(relevant_idx: int, retrieved_indices: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if relevant_idx in retrieved_indices:
        rank = retrieved_indices.index(relevant_idx) + 1 # 1-based index
        return 1.0 / rank
    return 0.0

def calculate_recall(relevant_idx: int, retrieved_indices: List[int]) -> float:
    """Calculate Recall@5."""
    return 1.0 if relevant_idx in retrieved_indices else 0.0

def process_recipe(recipe_idx: str, 
                   recipe_data: Dict[str, Any],
                   title_to_idx: Dict[str, int],
                   model: SentenceTransformer, 
                   index: faiss.IndexFlatL2) -> Dict[str, float]:
    """Process a single recipe and return metrics."""
    mrr_scores = []
    recall_scores = []
    retrieval_times = []

    relevant_idx = title_to_idx[recipe_data['title']]

    for prompt in recipe_data['prompts']:
        start_time = time.time()
        query_embedding = model.encode([prompt])
        retrieved_indices = retrieve_recipes_faiss(index, query_embedding)
        retrieval_time = time.time() - start_time

        mrr = calculate_mrr(relevant_idx, retrieved_indices)
        recall = calculate_recall(relevant_idx, retrieved_indices)

        mrr_scores.append(mrr)
        recall_scores.append(recall)
        retrieval_times.append(retrieval_time)

    return {
        "mrr": np.mean(mrr_scores),
        "recall": np.mean(recall_scores),
        "avg_retrieval_time": np.mean(retrieval_times),
    }

def run_experiment(df: pd.DataFrame, 
                   prompts: Dict[str, Dict[str, Any]], 
                   sample_sizes: List[int], 
                   embedding_models: List[str]) -> pd.DataFrame:
    """Run the main experiment and return results as a DataFrame."""
    print("Starting the experiment...")
    results = []

    all_recipe_titles = list(prompts.keys())
    selected_titles = []

    for sample_size in tqdm(sample_sizes, desc="Sample sizes"):
        # Add new random titles to reach the current sample size
        new_titles = random.sample([t for t in all_recipe_titles if t not in selected_titles], 
                                   sample_size - len(selected_titles))
        selected_titles.extend(new_titles)

        sample_df = df[df['Title'].isin([prompts[t]['title'] for t in selected_titles])]
        title_to_idx = {title: idx for idx, title in enumerate(sample_df['Title'])}

        for embedding_model_name in tqdm(embedding_models, desc="Embedding models", leave=False):
            print(f"Using embedding model: {embedding_model_name}")
            model = SentenceTransformer(embedding_model_name)

            print("Embedding recipes...")
            start_time = time.time()
            embeddings = embed_recipes(sample_df, model)
            embedding_time = time.time() - start_time
            print(f"Embedding time: {embedding_time:.2f} seconds")

            print("Creating FAISS index...")
            start_time = time.time()
            index = create_faiss_index(embeddings)
            indexing_time = time.time() - start_time
            print(f"Indexing time: {indexing_time:.2f} seconds")

            print("Retrieving recipes and calculating metrics...")
            with ThreadPoolExecutor() as executor:
                futures = []
                for recipe_idx in selected_titles:
                    futures.append(executor.submit(process_recipe, recipe_idx, prompts[recipe_idx], title_to_idx, model, index))

                mrr_scores = []
                recall_scores = []
                retrieval_times = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing recipes", leave=False):
                    result = future.result()
                    mrr_scores.append(result["mrr"])
                    recall_scores.append(result["recall"])
                    retrieval_times.append(result["avg_retrieval_time"])

            mean_mrr = np.mean(mrr_scores)
            mean_recall = np.mean(recall_scores)
            avg_retrieval_time = np.mean(retrieval_times)
            print(f"Mean MRR: {mean_mrr:.4f}")
            print(f"Mean Recall@5: {mean_recall:.4f}")
            print(f"Average retrieval time: {avg_retrieval_time:.4f} seconds")

            results.append({
                "sample_size": sample_size,
                "embedding_model": embedding_model_name,
                "mean_mrr": mean_mrr,
                "mean_recall": mean_recall,
                "avg_retrieval_time": avg_retrieval_time,
                "embedding_time": embedding_time,
                "indexing_time": indexing_time
            })

    return pd.DataFrame(results)

def create_comparison_chart(df: pd.DataFrame, x: str, y: str, title: str, filename: str):
    """Create and save a comparison chart."""
    plt.figure(figsize=(10, 6))
    for model in df['embedding_model'].unique():
        model_data = df[df['embedding_model'] == model]
        plt.plot(model_data[x], model_data[y], marker='o', label=model)
    
    plt.xlabel(x.capitalize())
    plt.ylabel(y.upper())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    df = load_data('./data/recipe_embeddings.pkl')
    prompts = load_prompts('./data/generated_prompts.json')
    sample_sizes = [100, 200]
    embedding_models = ['all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1']
    
    results_df = run_experiment(df, prompts, sample_sizes, embedding_models)
    
    # Save results to CSV
    results_df.to_csv('experiment_results.csv', index=False)
    print("Results saved to experiment_results.csv")

    # Create and save charts
    create_comparison_chart(results_df, 'sample_size', 'mean_mrr', 
                            'MRR Comparison for Different Embedding Models', 
                            'mrr_comparison.png')
    print("MRR comparison chart saved to mrr_comparison.png")

    create_comparison_chart(results_df, 'sample_size', 'mean_recall', 
                            'Recall@5 Comparison for Different Embedding Models', 
                            'recall_comparison.png')
    print("Recall comparison chart saved to recall_comparison.png")

if __name__ == "__main__":
    main()
    
    # example of retrieving the results
    # filter out df for recipes from prompts based on title
    df_1000 = df[df['Title'].isin([prompts[t]['title'] for t in prompts.keys()])]
    # embed the recipes
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_1000 = embed_recipes(df_1000, model)
    # create faiss index
    index_1000 = create_faiss_index(embeddings_1000)
    # calculate metrics for full prompt file
    results_1000 = []
    mrr = []
    recall = []
    retrieval_time = []
    for recipe_idx in prompts.keys():
        title_to_idx = {title: idx for idx, title in enumerate(df_1000['Title'])}
        result = process_recipe(recipe_idx, prompts[recipe_idx], title_to_idx, model, index_1000)
        # Store title 
        results_1000.append(result)
        mrr.append(result["mrr"])
        recall.append(result["recall"])
        retrieval_time.append(result["avg_retrieval_time"])
    
    # show historgram distribution of mrr with labels and values on the bars
    plt.hist(mrr, bins=20, color='blue', alpha=0.7)
    plt.axvline(np.mean(mrr), color='red', linestyle='dashed', linewidth=1)
    plt.xlabel('MRR')
    plt.ylabel('Frequency')
    plt.title('Mean Reciprocal Rank Distribution')
    plt.show()
    
    # Show recall distribution
    plt.hist(recall, bins=20, color='green', alpha=0.7)
    plt.axvline(np.mean(recall), color='red', linestyle='dashed', linewidth=1)
    plt.xlabel('Recall@5')
    plt.ylabel('Frequency')
    plt.title('Recall@5 Distribution')
    
    # Find the recipe with the lowest MR
    
    # convert first row of df_1000 without embedding to dictionary
    df_1000.columns
    