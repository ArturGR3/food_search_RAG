"""
This script compares the performance of FAISS and LanceDB for vector search and retrieval.
It evaluates different search methods using metrics like MRR (Mean Reciprocal Rank) and Recall.
"""

import pandas as pd
import numpy as np
import faiss
import time
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import lancedb

def data_pull():
    """
    Pull data from pickle and CSV files.
    
    Returns:
        tuple: A tuple containing the dataframe and prompts.
    """
    df = pd.read_pickle("./data/recipe_embeddings.pkl")['df']
    prompts = pd.read_csv('./data/prompts_dataframe.csv')
    return df, prompts

def data_embedding(df, prompts, sample_size=10, model_name="multi-qa-MiniLM-L6-cos-v1"):
    """
    Embed data using a sentence transformer model.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        prompts (pd.DataFrame): Prompts dataframe.
        sample_size (int): Number of samples to use.
        model_name (str): Name of the sentence transformer model.
    
    Returns:
        tuple: A tuple containing the sample dataframe and embeddings.
    """
    sample_prompts = prompts.sample(n=sample_size, random_state=42, replace=False)
    sample_df = pd.merge(sample_prompts, df, on='index')
    
    model = get_registry().get("sentence-transformers").create(name=model_name, device="cpu", normalize=True)
    text_for_embedding_embeddings = np.array(model.compute_query_embeddings(sample_df['text_for_embedding'].tolist()), dtype=np.float16)
    
    all_prompts = [p for prompt_list in sample_df['prompts'].apply(eval).tolist() for p in prompt_list]
    all_prompts_embeddings = np.array(model.compute_query_embeddings(all_prompts), dtype=np.float16)
    
    prompts_embeddings = []
    index = 0
    for prompt_list in sample_df['prompts'].apply(eval).tolist():
        num_prompts = len(prompt_list)
        prompts_embeddings.append(all_prompts_embeddings[index:index + num_prompts])
        index += num_prompts
    
    sample_df['prompts_embeddings'] = prompts_embeddings
    return sample_df, text_for_embedding_embeddings

def create_faiss_index(embeddings):
    """
    Create a FAISS index for the given embeddings.
    
    Args:
        embeddings (np.array): Array of embeddings.
    
    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    return faiss_index

def faiss_search(faiss_index, sample_df, k=5):
    """
    Perform search using FAISS index.
    
    Args:
        faiss_index (faiss.IndexFlatL2): FAISS index.
        sample_df (pd.DataFrame): Sample dataframe.
        k (int): Number of top results to retrieve.
    
    Returns:
        tuple: A tuple containing search results and retrieval time.
    """
    results = []
    start_time = time.time()
    for i in range(sample_df.shape[0]):
        prompts = eval(sample_df.iloc[i]['prompts'])
        prompts_embeddings = sample_df.iloc[i]['prompts_embeddings']
        for prompt, prompt_embedding in zip(prompts, prompts_embeddings):
            D, I = faiss_index.search(prompt_embedding.reshape(1, -1), k=k)
            retrieved = sample_df.iloc[I[0]]
            results.append({
                'ground_truth': {'index': sample_df.iloc[i]['index'], 'title': sample_df.iloc[i]['title']},
                'retrieved': [{'index': retrieved.iloc[j]['index'], 'title': retrieved.iloc[j]['title']} for j in range(k)]
            })
    retrieval_time = time.time() - start_time
    return results, retrieval_time

def lancedb_search(sample_df, text_for_embedding_embeddings, search_type='vector', top_k=2, model_name="multi-qa-MiniLM-L6-cos-v1"):
    """
    Perform search using LanceDB.
    
    Args:
        sample_df (pd.DataFrame): Sample dataframe.
        text_for_embedding_embeddings (np.array): Array of embeddings.
        search_type (str): Type of search ('vector', 'fts', or 'hybrid').
        top_k (int): Number of top results to retrieve.
        model_name (str): Name of the sentence transformer model.
    
    Returns:
        tuple: A tuple containing search results and retrieval time.
    """
    model = get_registry().get("sentence-transformers").create(name=model_name, device="cpu", normalize=True)
    
    class Recipes(LanceModel):
        index: int
        title: str
        text_for_embedding: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()
    
    db = lancedb.connect("./lancedb")
    table = db.create_table("recipes", mode="overwrite", schema=Recipes)
    
    data = sample_df.drop(columns=['prompts_embeddings']).copy() 
    data['vector'] = text_for_embedding_embeddings.tolist()
    table.add(data[['index', 'title', 'text_for_embedding', 'vector']])
    table.create_fts_index("text_for_embedding", use_tantivy=False)
    
    results = []
    start_time = time.time()
    for i in range(sample_df.shape[0]):
        prompts = eval(sample_df.iloc[i]['prompts'])
        prompts_embeddings = sample_df.iloc[i]['prompts_embeddings']
        for prompt, prompt_embedding in zip(prompts, prompts_embeddings):
            if search_type == 'vector':
                search_results = table.search(prompt_embedding, vector_column_name="vector", query_type=search_type).\
                select(["index", "title"]). \
                limit(top_k).\
                to_list()
            elif search_type == 'fts':
                search_results = table.search(prompt, query_type=search_type). \
                select(["index", "title"]). \
                limit(top_k). \
                to_list()
            elif search_type == 'hybrid':
                search_results = table.search(query = prompt, query_type=search_type). \
                select(["index", "title"]). \
                limit(top_k). \
                to_list()
            results.append({
                'ground_truth': {'index': sample_df.iloc[i]['index'], 'title': sample_df.iloc[i]['title']},
                'retrieved': [{'index': res['index'], 'title': res['title']} for res in search_results]
            })
    retrieval_time = time.time() - start_time
    return results, retrieval_time

def mrr_at_k(results, k):
    """
    Calculate Mean Reciprocal Rank at k.
    
    Args:
        results (list): List of search results.
        k (int): Rank to calculate MRR at.
    
    Returns:
        float: MRR@k value.
    """
    mrr = 0.0
    for result in results:
        ground_truth_index = result['ground_truth']['index']
        for rank, item in enumerate(result['retrieved'][:k]):
            if item['index'] == ground_truth_index:
                mrr += 1.0 / (rank + 1)
                break
    return mrr / len(results)

def recall_at_k(results, k):
    """
    Calculate Recall at k.
    
    Args:
        results (list): List of search results.
        k (int): Rank to calculate Recall at.
    
    Returns:
        float: Recall@k value.
    """
    recall = 0.0
    for result in results:
        ground_truth_index = result['ground_truth']['index']
        retrieved_indices = [item['index'] for item in result['retrieved'][:k]]
        if ground_truth_index in retrieved_indices:
            recall += 1.0
    return recall / len(results)

def main(sample_size=10):
    """
    Main function to run the comparison between FAISS and LanceDB.
    """
    df, prompts = data_pull()
    sample_df, text_for_embedding_embeddings = data_embedding(df, prompts, sample_size=sample_size)
    faiss_index = create_faiss_index(text_for_embedding_embeddings)

    faiss_results, faiss_retrieval_time = faiss_search(faiss_index, sample_df, k=5)
    lancedb_results, lancedb_retrieval_time = lancedb_search(sample_df, text_for_embedding_embeddings, search_type='vector', top_k=5)
    lancedb_results_fts, lancedb_fts_retrieval_time = lancedb_search(sample_df, text_for_embedding_embeddings, search_type='fts', top_k=5)
    lancedb_results_hybrid, lancedb_hybrid_retrieval_time = lancedb_search(sample_df, text_for_embedding_embeddings, search_type='hybrid', top_k=5)

    # Calculate metrics
    metrics = {
        'FAISS': {
            'MRR@1': mrr_at_k(faiss_results, k=1),
            'MRR@3': mrr_at_k(faiss_results, k=3),
            'MRR@5': mrr_at_k(faiss_results, k=5),
            'Recall@1': recall_at_k(faiss_results, k=1),
            'Recall@3': recall_at_k(faiss_results, k=3),
            'Recall@5': recall_at_k(faiss_results, k=5),
            'Retrieval Time': faiss_retrieval_time
        },
        'LanceDB Vector': {
            'MRR@1': mrr_at_k(lancedb_results, k=1),
            'MRR@3': mrr_at_k(lancedb_results, k=3),
            'MRR@5': mrr_at_k(lancedb_results, k=5),
            'Recall@1': recall_at_k(lancedb_results, k=1),
            'Recall@3': recall_at_k(lancedb_results, k=3),
            'Recall@5': recall_at_k(lancedb_results, k=5),
            'Retrieval Time': lancedb_retrieval_time
        },
        'LanceDB FTS (text search)': {
            'MRR@1': mrr_at_k(lancedb_results_fts, k=1),
            'MRR@3': mrr_at_k(lancedb_results_fts, k=3),
            'MRR@5': mrr_at_k(lancedb_results_fts, k=5),
            'Recall@1': recall_at_k(lancedb_results_fts, k=1),
            'Recall@3': recall_at_k(lancedb_results_fts, k=3),
            'Recall@5': recall_at_k(lancedb_results_fts, k=5),
            'Retrieval Time': lancedb_fts_retrieval_time
        },
        'LanceDB Hybrid': {
            'MRR@1': mrr_at_k(lancedb_results_hybrid, k=1),
            'MRR@3': mrr_at_k(lancedb_results_hybrid, k=3),
            'MRR@5': mrr_at_k(lancedb_results_hybrid, k=5),
            'Recall@1': recall_at_k(lancedb_results_hybrid, k=1),
            'Recall@3': recall_at_k(lancedb_results_hybrid, k=3),
            'Recall@5': recall_at_k(lancedb_results_hybrid, k=5),
            'Retrieval Time': lancedb_hybrid_retrieval_time
        }
    }

    # Find the best method for each metric
    best_methods = {}
    for metric in ['MRR@1', 'MRR@3', 'MRR@5', 'Recall@1', 'Recall@3', 'Recall@5', 'Retrieval Time']:
        if metric == 'Retrieval Time':
            best_methods[metric] = min(metrics.items(), key=lambda x: x[1][metric])[0]
        else:
            best_methods[metric] = max(metrics.items(), key=lambda x: x[1][metric])[0]

    # Prepare data for tabulation
    table_data = []
    for method, results in metrics.items():
        row = [
            method,
            f"{results['MRR@1']:.2f}" + (' *' if method == best_methods['MRR@1'] else ''),
            f"{results['MRR@3']:.2f}" + (' *' if method == best_methods['MRR@3'] else ''),
            f"{results['MRR@5']:.2f}" + (' *' if method == best_methods['MRR@5'] else ''),
            f"{results['Recall@1']:.2f}" + (' *' if method == best_methods['Recall@1'] else ''),
            f"{results['Recall@3']:.2f}" + (' *' if method == best_methods['Recall@3'] else ''),
            f"{results['Recall@5']:.2f}" + (' *' if method == best_methods['Recall@5'] else ''),
            f"{results['Retrieval Time']:.2f} s" + (' *' if method == best_methods['Retrieval Time'] else '')
        ]
        table_data.append(row)

    # Create and print the table
    headers = ['Method', 'MRR@1', 'MRR@3', 'MRR@5', 'Recall@1', 'Recall@3', 'Recall@5', 'Retrieval Time']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("\n* indicates the best performing method for each metric")
    
    # Save the table as a txt file
    with open('retrieval_comparison_results.txt', 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write("\n* indicates the best performing method for each metric")

if __name__ == "__main__":
    main(sample_size=500)