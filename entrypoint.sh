#!/bin/bash
set -e

if [[ "$1" == "api" ]]; then
    echo "Starting FastAPI server..."
    exec uvicorn api:app --host 0.0.0.0 --port 8000
elif [[ "$1" == "train" ]]; then
    echo "Starting model training..."
    exec python train.py --model "$2" --data "$3" --output "$4"
elif [[ "$1" == "test" ]]; then
    echo "Starting model testing..."
    exec python test.py --model "$2" --testfile "$3" --saved_model "$4"
else
    echo "Invalid argument. Use 'api', 'train', or 'test'."
    exit 1
fi

