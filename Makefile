# Makefile for Recipe Recommendation System

# Variables
IMAGE_NAME = recipe-recommendation-system
CONTAINER_NAME = recipe-app
HOST_PORT = 7861
CONTAINER_PORT = 7860
DATA_DIR = data

# Default target
.PHONY: all
all: build run

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

# View Docker logs
.PHONY: logs
logs:
	@echo "Viewing Docker logs..."
	docker logs -f $(CONTAINER_NAME)

# Clean up: stop container, remove image, and prune system
.PHONY: clean
clean:
	@echo "Stopping Docker container..."
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	@echo "Removing Docker image..."
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Pruning Docker system..."
	docker system prune -f

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make build        - Build the Docker image"
	@echo "  make run          - Run the Docker container"
	@echo "  make logs         - View Docker logs"
	@echo "  make clean        - Clean up Docker resources"
	@echo "  make help         - Show this help message"