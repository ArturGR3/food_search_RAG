# Makefile for Recipe Recommendation System

# Variables
IMAGE_NAME = recipe-recommendation-system
CONTAINER_NAME = recipe-app
HOST_PORT = 7861# this is the port that the app runs on in the host
CONTAINER_PORT = 7860# this is the port that the app runs on in the container
DATA_DIR = data
KAGGLE_DATASET = pes12017000148/food-ingredients-and-recipe-dataset-with-images

# Default target
.PHONY: all
all: setup preprocess build run

# Setup: Install required tools and download data
.PHONY: setup
setup:
	@echo "Setting up environment..."
	pip install kaggle
	sudo apt-get install -y unzip
	mkdir -p $(DATA_DIR)

# Download and preprocess data
.PHONY: preprocess
preprocess: download_data process_data move_images

# Download data from Kaggle
.PHONY: download_data
download_data:
	@echo "Downloading dataset from Kaggle..."
	kaggle datasets download -d $(KAGGLE_DATASET) -p $(DATA_DIR) && \
	unzip $(DATA_DIR)/food-ingredients-and-recipe-dataset-with-images.zip -d $(DATA_DIR) && \
	rm $(DATA_DIR)/food-ingredients-and-recipe-dataset-with-images.zip

# Process data using the Python script
.PHONY: process_data
process_data:
	@echo "Processing data..."
	python src/data_preprocessing/data_preprocessing.py

# Move images to the correct directory
.PHONY: move_images
move_images:
	@echo "Moving images to Food_Images directory..."
	mkdir -p $(DATA_DIR)/Food_Images
	mv $(DATA_DIR)/Food\ Images/Food\ Images/* $(DATA_DIR)/Food_Images/

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
.PHONY: run
run:
	@echo "Running Docker container..."
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker run -d --name $(CONTAINER_NAME) -p $(HOST_PORT):$(CONTAINER_PORT) $(IMAGE_NAME)
	@echo "Application is running at http://localhost:$(HOST_PORT)"

# Stop and remove the Docker container
.PHONY: stop
stop:
	@echo "Stopping and removing Docker container..."
	docker stop -f $(CONTAINER_NAME) 2>/dev/null || true
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true

# View Docker logs
.PHONY: logs
logs:
	@echo "Viewing Docker logs..."
	docker logs $(CONTAINER_NAME)

# Enter the Docker container
.PHONY: shell
shell:
	@echo "Entering Docker container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Clean up: stop container, remove image, and prune system
.PHONY: clean
clean: stop
	@echo "Removing Docker image..."
	docker rmi $(IMAGE_NAME)
	@echo "Pruning Docker system..."
	docker system prune -f

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make setup        - Install required tools and create data directory"
	@echo "  make preprocess   - Download and preprocess data"
	@echo "  make build        - Build the Docker image"
	@echo "  make run          - Run the Docker container"
	@echo "  make stop         - Stop and remove the Docker container"
	@echo "  make logs         - View Docker logs"
	@echo "  make shell        - Enter the Docker container"
	@echo "  make clean        - Clean up Docker resources"
	@echo "  make help         - Show this help message"