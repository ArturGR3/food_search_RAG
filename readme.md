# Recipe Recommendation System

This is a recipe recommendation system that suggests dishes based on user queries. It uses a combination of natural language processing and similarity search to find recipes that match the user's input.

## Table of Contents
- [Project Structure](#project-structure)
  - [Data Collection and Preprocessing](#1-data-collection-and-preprocessing)
  - [Generating Synthetic Data](#2-generating-synthetic-data-to-evaluate-retrieval-methods)
  - [Retrieval Method Comparison](#3-retrieval-method-comparison)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Data](#data)

## Project Structure

### 1. Data Collection and Preprocessing

The data collection and preprocessing steps are handled in two stages:

1. Data Collection:
   We use the Kaggle API to download the recipe dataset from a Kaggle competition [1]. The dataset contains food ingredients and recipes with image name mappings, providing a rich source of culinary information for our recipe recommendation system. Here's how to download the data:

   a. Ensure you have the Kaggle API credentials set up:
      - Go to your Kaggle account settings (https://www.kaggle.com/account)
      - Scroll down to the "API" section and click "Create New API Token"
      - This will download a `kaggle.json` file. Place this file in `~/.kaggle/` directory (create it if it doesn't exist)
      - Run `chmod 600 ~/.kaggle/kaggle.json` to set the correct permissions

   b. Install the Kaggle API and `unzip` utility if you haven't already:   
      ```bash
      pip install kaggle
      sudo apt-get install unzip
      ```

   c. Run the following commands to download, extract the dataset, and then delete the zip file:
      ```bash
      kaggle datasets download -d pes12017000148/food-ingredients-and-recipe-dataset-with-images -p ../data && \ 
          unzip ../data/food-ingredients-and-recipe-dataset-with-images.zip -d ../data && \ 
          rm ../data/food-ingredients-and-recipe-dataset-with-images.zip
      ```

   These commands download the dataset to the `../data` directory, extract its contents into the same directory, and then remove the zip file to save space.

2. Data Preprocessing:
   After downloading and extracting the data, we use [data_preprocessing.py](./src/data_preprocessing/data_preprocessing.py) to process the raw data. This script performs the following tasks:
   - Loads the downloaded CSV file
   - Combines the title, ingredients, and instructions into a single text for keyword search
   - Preprocesses the data for use in our recipe recommendation system
   - Saves the preprocessed data as `recipes.csv` for further use

3. Based on the findings in the retrieval comparison, we plan to use the LanceDB Full-Text Search (FTS) method for our recipe retrieval system in the future. It offers the best balance between accuracy (highest MRR and competitive Recall) and reasonable retrieval time. However, this feature is not yet implemented.

4. The preprocessed data is saved as `recipes.csv` for use in the current version of the system.
 
[1]: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images 

### 2. Generating Synthetic Data to Evaluate Retrieval Methods

To evaluate the initial retrieval performance of our system, we generate synthetic data in the form of user-like prompts for each recipe. This process is handled by the [ground_truth_generation.py](./src/data_preprocessing/ground_truth_generation.py) script.

1. Synthetic Data Generation:
   - We use **GPT-4-mini**, accessed through the OpenAI API, to generate 5 synthetic user prompts for each recipe.
   - Each prompt is designed to be a short, natural-sounding query (up to 10 words) that a user might use to search for the recipe.
   - This approach provides us with a diverse set of potential user inputs to test our retrieval system.

2. Asynchronous Processing with **asyncio**:
   - We utilize Python's asyncio library to handle concurrent API calls efficiently. asyncio allows us to run multiple I/O-bound tasks concurrently, significantly speeding up the data generation process.
   - Uses **asyncio.gather()** for concurrent API calls, with a semaphore to prevent rate limiting.

3. Implementation Details and Usage:
   - Supports processing test or full datasets:
     - Test mode (default): 10 samples, saves as `prompts_dataframe_test.csv`
     - Full dataset: 1000 samples, saves as `prompts_dataframe.csv`
   - Custom sample size possible with `--samples` argument

   Usage examples:
   ```bash
   python src/data_preprocessing/ground_truth_generation.py                    # Test mode (10 samples)
   python src/data_preprocessing/ground_truth_generation.py --samples 1000     # Full dataset (1000 samples)
   python src/data_preprocessing/ground_truth_generation.py --samples <number> # Custom number of samples
   ```

### 3. Retrieval Method Comparison

We compared the performance of FAISS and LanceDB for vector search and retrieval in the context of our Retrieval-Augmented Generation (RAG) application for food recipes. The evaluation uses metrics like Mean Reciprocal Rank (MRR) and Recall for top 1, 3, and 5 results.

#### Context

This evaluation is part of our larger RAG application where users search for food recipes. The dataset consists of LLM-generated prompts for given recipes. The evaluation checks if a given search engine can retrieve the original recipe that these prompts are based on, simulating real-world user queries.

#### Search Engines

1. FAISS (Facebook AI Similarity Search):
   FAISS is a library for efficient similarity search and clustering of dense vectors. It's optimized for speed and memory usage, making it suitable for large-scale vector search tasks.

2. LanceDB:
   LanceDB is a vector database that supports multiple search types:
   - Vector Search: Semantic search using vector embeddings
   - Full-Text Search (FTS): Traditional keyword-based text search
   - Hybrid Search: Combines vector and text search for potentially improved results

#### Evaluation Process

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

#### Results

The detailed results of the comparison (sample size: 500) can be found in the [retrieval results](src/retrieval/retrieval_comparison_results.txt) file. Here's a summary of the key findings:
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

#### Key Findings:
1. FAISS has the fastest retrieval time.
2. LanceDB FTS (text search) performs best in terms of MRR@1, MRR@3, MRR@5, and Recall@1.
3. LanceDB Hybrid search achieves the highest Recall@3 and Recall@5, but at the cost of significantly longer retrieval time.
4. LanceDB Vector search and FAISS show identical performance in terms of MRR and Recall, but FAISS is faster.



#### Conclusion

Based on these results, we have decided to proceed with the LanceDB Full-Text Search (FTS) method for our recipe retrieval system. It offers the best balance between accuracy (highest MRR and competitive Recall) and reasonable retrieval time. While the hybrid search shows slightly better Recall@3 and Recall@5, the significant increase in retrieval time doesn't justify its use for our current application needs.

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

The recipe data with images are stored in `recipes.csv`
The embeddings are stored in `embeddings.npy`.
The FAISS index for similarity search is stored in `recipe_index.faiss`.