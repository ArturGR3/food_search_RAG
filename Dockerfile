# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files and directories
COPY app.py /app/
COPY recipe_recommendation.py /app/
COPY requirements.txt /app/
COPY src/ /app/src/
COPY data/recipes.csv /app/data/
COPY .env_example /app/.env  

# Create the Food_Images directory
RUN mkdir -p /app/data/Food_Images

# Copy the images to the Food_Images directory
COPY data/Food_Images/ /app/data/Food_Images/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]