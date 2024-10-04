# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files and directories
COPY app.py /app/
COPY recipe_recommendation.py /app/
COPY requirements.txt /app/
COPY src/ /app/src/
COPY .env /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/Food_Images

# Run data setup and preprocessing
RUN python /app/src/data_preprocessing/setup_data.py && \
    python /app/src/data_preprocessing/data_preprocessing.py

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]
