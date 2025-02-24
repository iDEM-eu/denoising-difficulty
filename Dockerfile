# Use official lightweight Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to avoid unwanted warnings
ENV TOKENIZERS_PARALLELISM=false

# Allow passing arguments to pick a model
ENTRYPOINT ["python", "run_training.py"]
