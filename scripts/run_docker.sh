#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.pytorch -t chatbot-pytorch .

# Run the container with GPU support
echo "Running container with GPU support..."
docker run --gpus all -it \
    -p 8888:8888 \
    -v $(pwd):/app \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/Dataset:/app/Dataset \
    --name chatbot-container \
    chatbot-pytorch

echo "Container started! You can now run your Python scripts inside the container."
echo "To access Jupyter notebook, run: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root" 