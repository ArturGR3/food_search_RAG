## This script brings data from kaggle to the local machine 
import os
import zipfile
import shutil
import time 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Explicitly set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
    
#sleep for 10 seconds
time.sleep(5)

from kaggle.api.kaggle_api_extended import KaggleApi

def setup_data():
    # Load environment variable
    
    

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Download dataset
    dataset = "pes12017000148/food-ingredients-and-recipe-dataset-with-images"
    api.dataset_download_files(dataset, path=data_dir)
    print("Dataset downloaded successfully.")

    # Unzip the dataset
    zip_file = os.path.join(data_dir, 'food-ingredients-and-recipe-dataset-with-images.zip')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Dataset unzipped successfully.")

    # Remove the zip file
    os.remove(zip_file)
    print("Zip file removed successfully.")

    # Move images to the correct directory
    source_dir = os.path.join(data_dir, 'Food Images', 'Food Images')
    dest_dir = os.path.join(data_dir, 'Food_Images')
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(dest_dir, item)
        shutil.move(s, d)
    print("Images moved to the correct directory successfully.")

    # Remove the empty directories
    shutil.rmtree(os.path.join(data_dir, 'Food Images'))
    print("Empty directories removed successfully.")

    print("Data setup completed successfully.")

if __name__ == "__main__":
    setup_data()