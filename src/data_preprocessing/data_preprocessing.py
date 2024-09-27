# Description: This script preprocesses the recipe dataset and prepares it for search.

import pandas as pd
import os
import ast

# Define constants
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
CSV_FILENAME = 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
OUTPUT_FILENAME = 'recipes.csv'

# Ensure data directory exists
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"Created data directory at {DATA_PATH}")

def load_data():
    csv_path = os.path.join(DATA_PATH, CSV_FILENAME)
    print(csv_path)
    if os.path.exists(csv_path):
        print('Data exists. Loading data...')
        return pd.read_csv(csv_path)
    else:
        print('Data does not exist. Please ensure the CSV file is in the correct location.')
        return None

def preprocess_data(df):
    # Create a unique index for each title based on row number
    df['Index'] = df.index

    df['text_for_search'] = df.apply(lambda row: (
        f"{row['Title']}. "
        f"Ingredients: {', '.join(ast.literal_eval(row['Cleaned_Ingredients']))}. "
        f"Instructions: {row['Instructions']}"
    ), axis=1)
    return df

def main():
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        df.to_csv(os.path.join(DATA_PATH, OUTPUT_FILENAME), index=False)
        print(f"Preprocessed data saved to {OUTPUT_FILENAME}")
    else:
        print("Data preprocessing failed due to missing input file.")

if __name__ == "__main__":
    main()
