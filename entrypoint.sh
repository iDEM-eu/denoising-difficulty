#!/bin/bash

set -e

# Run training with the specified model
if [ -z "$1" ]; then
    echo "No model specified. Use: docker run model-training <model_name>"
    exit 1
fi

MODEL=$1
echo "Starting training for model: $MODEL"

# Run the training script
python run-training.py --model "$MODEL"
