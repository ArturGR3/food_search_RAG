# Recipe Recommendation System

This is a recipe recommendation system that suggests dishes based on user queries. It uses a combination of natural language processing and similarity search to find recipes that match the user's input.

## The structure of the project

### 1. Data collection and preprocessing

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

3. Based on the findings in the [retrieval comparison](./src/retrieval/retrieval_readme.md), we plan to use the LanceDB Full-Text Search (FTS) method for our recipe retrieval system in the future. It offers the best balance between accuracy (highest MRR and competitive Recall) and reasonable retrieval time. However, this feature is not yet implemented.

4. The preprocessed data is saved as `recipes.csv` for use in the current version of the system.
 
[1]: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images 

### 2. Generating synthetic data to evaluate retrieval methods

To evaluate the initial retrieval performance of our system, we generate synthetic data in the form of user-like prompts for each recipe. This process is handled by the [ground_truth_generation.py](./src/data_preprocessing/ground_truth_generation.py) script.

1. Synthetic Data Generation:
   - We use **GPT-4-mini**, accessed through the OpenAI API, to generate 5 synthetic user prompts for each recipe.
   - Each prompt is designed to be a short, natural-sounding query (up to 10 words) that a user might use to search for the recipe.
   - This approach provides us with a diverse set of potential user inputs to test our retrieval system.

2. Asynchronous Processing with **asyncio**:
   - We utilize Python's asyncio library to handle concurrent API calls efficiently.
   - asyncio allows us to run multiple I/O-bound tasks concurrently, significantly speeding up the data generation process.

3. Implementation Details:
   - We use **asyncio.gather()** to run multiple API calls concurrently.
   - A semaphore is implemented to limit the number of concurrent API calls, preventing rate limiting issues and ensuring efficient use of resources.
   - The script supports processing either a sample dataset or the full dataset:
     - By default, it processes 10 samples and saves the result as `prompts_dataframe_test.csv`.
     - For the full dataset (1000 samples), use the `--samples 1000` flag, which saves the result as `prompts_dataframe.csv`.
   - You can specify any number of samples using the `--samples` argument.

4. Usage:
   - For the default test dataset (10 samples):
     ```
     python src/data_preprocessing/ground_truth_generation.py
     ```
   - For the full dataset (1000 samples):
     ```
     python src/data_preprocessing/ground_truth_generation.py --samples 1000
     ```
   - For a custom number of samples:
     ```
     python src/data_preprocessing/ground_truth_generation.py --samples <number>
     ```

5. Output:
   - The generated prompts are saved in a CSV file (`prompts_dataframe_test.csv` for samples, `prompts_dataframe.csv` for the full dataset) for further use in evaluating the retrieval system.

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

The recipe data with images are store in `recipes.csv`
The embeddings are stored in `embeddings.npy`.
The FAISS index for similarity search is stored in `recipe_index.faiss`.
