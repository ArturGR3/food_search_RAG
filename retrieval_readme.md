# Vector Search Performance Comparison: FAISS vs LanceDB for Recipe Retrieval

This step compares the performance of FAISS and LanceDB for vector search and retrieval in the context of a Retrieval-Augmented Generation (RAG) application for food recipes. It evaluates different search methods using metrics like Mean Reciprocal Rank (MRR) and Recall for top 1,3 and 5.

## Context

This evaluation is part of a larger RAG application where users search for food recipes. The dataset consists of LLM-generated prompts for given recipes. The evaluation checks if a given search engine can retrieve the original recipe that these prompts are based on, simulating real-world user queries.

## Overview

The script `comparison_retrieval.py` performs the following tasks:
1. Loads recipe data and synthetic prompts
2. Embeds the data using the sentence transformer model "multi-qa-MiniLM-L6-cos-v1"
3. Creates indexes for FAISS and LanceDB
4. Performs searches using different methods:
   - FAISS vector search
   - LanceDB vector search
   - LanceDB full-text search (FTS)
   - LanceDB hybrid search (vector + text)
5. Calculates performance metrics (MRR and Recall at k=1, k=3, and k=5)
6. Compares retrieval times

## Search Engines

### FAISS (Facebook AI Similarity Search)
FAISS is a library for efficient similarity search and clustering of dense vectors. It's optimized for speed and memory usage, making it suitable for large-scale vector search tasks.

### LanceDB
LanceDB is a vector database that supports multiple search types:
- Vector Search: Semantic search using vector embeddings
- Full-Text Search (FTS): Traditional keyword-based text search
- Hybrid Search: Combines vector and text search for potentially improved results

## Results

The detailed results of the comparison (sample size: 500) can be found in the [retrieval results](retrieval_comparison_results.txt) file. Here's a summary of the key findings:
```
+---------------------------+---------+---------+---------+------------+------------+------------+------------------+
| Method                    | MRR@1   | MRR@3   | MRR@5   | Recall@1   | Recall@3   | Recall@5   | Retrieval Time   |
+===========================+=========+=========+=========+============+============+============+==================+
| FAISS                     | 0.78    | 0.84    | 0.84    | 0.78       | 0.90       | 0.93       | 3.35 s *         |
+---------------------------+---------+---------+---------+------------+------------+------------+------------------+
| LanceDB Vector            | 0.78    | 0.84    | 0.84    | 0.78       | 0.90       | 0.93       | 9.70 s           |
+---------------------------+---------+---------+---------+------------+------------+------------+------------------+
| LanceDB FTS (text search) | 0.86 *  | 0.90 *  | 0.91 *  | 0.86 *     | 0.95       | 0.97       | 7.76 s           |
+---------------------------+---------+---------+---------+------------+------------+------------+------------------+
| LanceDB Hybrid            | 0.84    | 0.89    | 0.90    | 0.84       | 0.96 *     | 0.98 *     | 73.91 s          |
+---------------------------+---------+---------+---------+------------+------------+------------+------------------+
* indicates the best performing method for each metric
```
1. FAISS has the fastest retrieval time.
2. LanceDB FTS (text search) performs best in terms of MRR@1, MRR@3, MRR@5, and Recall@1.
3. LanceDB Hybrid search achieves the highest Recall@3 and Recall@5, but at the cost of significantly longer retrieval time.
4. LanceDB Vector search and FAISS show identical performance in terms of MRR and Recall, but FAISS is faster.

## Conclusion

Based on the results, we have decided to proceed with the LanceDB Full-Text Search (FTS) method for our recipe retrieval system. It offers the best balance between accuracy (highest MRR and competitive Recall) and reasonable retrieval time. While the hybrid search shows slightly better Recall@3 and Recall@5, the significant increase in retrieval time doesn't justify its use for our current application needs.
