import pandas as pd
import numpy as np
import pickle
import os
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load the existing data
with open('data/recipe_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    df = data['df']
    embeddings = data['embeddings']
    
    
# Add image data to the dataframe
image_folder = 'data/Food_Images'
df['Image_Data'] = df['Image_Name'].apply(lambda x: load_image(os.path.join(image_folder, f"{x}.jpg")) if os.path.exists(os.path.join(image_folder, f"{x}.jpg")) else None)

# Function to display an image from base64 string
def display_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Display the first image in the dataframe
first_image_data = df['Image_Data'].iloc[0]
print(f"Displaying image for recipe: {df['Title'].iloc[0]}")
display_image(first_image_data)


# Save the updated data
with open('data/recipe_embeddings_with_images.pkl', 'wb') as f:
    pickle.dump({'df': df, 'embeddings': embeddings}, f)

print("Updated recipe_embeddings.pkl with image data")


import pandas as pd
import numpy as np
import pickle

# Load the existing data
with open('data/recipe_embeddings_with_images.pkl', 'rb') as f:
    data = pickle.load(f)
    df = data['df']
    embeddings = data['embeddings']

# Save DataFrame to CSV
df.to_csv('recipes.csv', index=False)

# Save embeddings to .npy file
np.save('embeddings.npy', embeddings)

print("Data converted and saved in new formats.")