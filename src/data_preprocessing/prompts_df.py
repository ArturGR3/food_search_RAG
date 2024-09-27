import pandas as pd
import json

# Load the generated prompts JSON
with open('./data/generated_prompts.json', 'r') as f:
    prompts = json.load(f)

# Create a list of dictionaries with the desired structure
data = []
for index, value in prompts.items():
    data.append({
        'index': index,
        'title': value['title'],
        'prompts': value['prompts']
    })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv('./data/prompts_dataframe.csv', index=False)