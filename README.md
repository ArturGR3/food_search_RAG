# Recipe Recommendation System

This project contains a Recipe Recommendation System built with Python and Gradio.

## Project Structure

Before running the setup and preprocessing steps, your project structure should look like this:

```
.
├── app.py
├── recipe_recommendation.py
├── requirements.txt
├── Dockerfile
├── Makefile
├── README.md
├── .env
└── src/
    ├── reporting/
    │   └── db.py
    ├── utils/
    │   ├── llm_factory.py
    │   └── settings.py
    └── data_preprocessing/
        └── data_preprocessing.py
```

After running `make setup` and `make preprocess`, your project structure will include the downloaded and processed data:

```
.
├── app.py
├── recipe_recommendation.py
├── requirements.txt
├── Dockerfile
├── Makefile
├── README.md
├── .env
├── src/
│   ├── reporting/
│   │   └── db.py
│   ├── utils/
│   │   ├── llm_factory.py
│   │   └── settings.py
│   └── data_preprocessing/
│       └── data_preprocessing.py
└── data/
    ├── recipes.csv
    └── Food_Images/
        └── ... (recipe images)
```

## Environment Variables

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Replace `your_*_api_key` and Kaggle credentials with your actual keys.

## Setup and Running with Make

This project uses a Makefile to automate setup, preprocessing, and Docker operations. Here are the main commands:

1. Setup the environment:
   ```
   make setup
   ```
   This installs required tools like the Kaggle API and unzip utility.

2. Preprocess the data:
   ```
   make preprocess
   ```
   This downloads the dataset from Kaggle, processes it, and moves images to the correct directory.

3. Build the Docker image:
   ```
   make build
   ```

4. Run the Docker container:
   ```
   make run
   ```

5. To see all available commands:
   ```
   make help
   ```

## Running with Docker Manually

If you prefer to run Docker commands manually:

1. Make sure you have Docker installed on your system.

2. Build the Docker image:
   ```
   docker build -t recipe-recommendation-system .
   ```

3. Run the Docker container:
   ```
   docker run -p 7860:7860 recipe-recommendation-system
   ```

4. Open your web browser and go to `http://localhost:7860` to access the application.

## Notes

- The application runs on port 7860 inside the container, which is mapped to the same port on your host machine.
- Make sure all necessary files are in the correct directories before building the image.
- The Dockerfile will copy the images from `data/Food_Images/` into the container. Ensure your images are in this directory before building.
- The `.env` file is copied into the container to provide the necessary API keys.

## Troubleshooting

If you encounter any issues:
- Run `make setup` to ensure all required tools are installed.
- Run `make preprocess` to download and process the data correctly.
- Ensure all required files are present in the project directory.
- Check that the port 7860 is not being used by another application on your host machine.
- Verify that your images are correctly placed in the `data/Food_Images/` directory.
- Make sure your `.env` file contains valid API keys.
- Review the Docker logs using `make logs` or:
  ```
  docker logs <container-id>
  ```

For more information or if you encounter any problems, please refer to the project documentation or open an issue on the project repository.