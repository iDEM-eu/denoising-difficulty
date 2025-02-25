import argparse
import os
import json
import subprocess

# Load Configurations
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Parse arguments
parser = argparse.ArgumentParser(description="Run the correct training script based on the selected model.")
parser.add_argument("--model", type=str, required=True, help="Model name to train (st, baseline, label_smoothing, ntm)")
args = parser.parse_args()

MODEL_NAME = args.model
if MODEL_NAME not in config["models"]:
    raise ValueError(f"Invalid model name: {MODEL_NAME}. Available models: {list(config['models'].keys())}")

# Map models to their respective scripts
MODEL_SCRIPTS = {
    "st": "train_st.py",
    "baseline": "train_baseline.py",
    "label_smoothing": "train_label_smoothing.py",
    "ntm": "train_ntm.py"
}

# Pick the right script
script_to_run = MODEL_SCRIPTS[MODEL_NAME]
print(f"Running training for model: {MODEL_NAME} using {script_to_run}")

# Execute the script
subprocess.run(["python", script_to_run, "--config", CONFIG_FILE, "--model", MODEL_NAME])
