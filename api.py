import os
import json
import torch
import uvicorn
import wandb
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from utils.metrics_utils import compute_generalization_metrics, compute_metrics
from datasets import Dataset
import pandas as pd

# Load configuration
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

app = FastAPI(title="Model Training & Inference API")

# Initialize tokenizers and models dynamically
loaded_models = {}
loaded_tokenizers = {}

def load_model(model_name):
    """Load model and tokenizer from configuration"""
    if model_name not in config["models"]:
        raise ValueError(f"Model {model_name} not found in config.")
    
    model_config = config["models"][model_name]
    model_path = model_config["output_model_dir"]
    
    if model_name not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_model"])
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        loaded_models[model_name] = model
        loaded_tokenizers[model_name] = tokenizer
    
    return loaded_models[model_name], loaded_tokenizers[model_name]

# Request Schema for Inference
class InferenceRequest(BaseModel):
    model_name: str
    text: str

@app.post("/predict/")
def predict(request: InferenceRequest):
    """Run inference using the trained model"""
    try:
        model, tokenizer = load_model(request.model_name)
        inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).tolist()[0]
            prediction = torch.argmax(logits, dim=-1).item()

        return {"prediction": prediction, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Request Schema for Training
class TrainRequest(BaseModel):
    model_name: str

@app.post("/train/")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a model asynchronously"""
    model_name = request.model_name
    if model_name not in config["models"]:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found in config.")

    def train():
        """Train the model in the background"""
        model_config = config["models"][model_name]
        train_data_path = model_config["dataset_path"]
        
        wandb.init(project=model_config["wandb_project"], name=model_name, reinit=True)

        # Load dataset
        df = pd.read_csv(train_data_path, sep=model_config["separator"])
        train_dataset = Dataset.from_pandas(df)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_model"])
        model = AutoModelForSequenceClassification.from_pretrained(model_config["tokenizer_model"], num_labels=2)

        def preprocess_function(examples):
            return tokenizer(examples[model_config["text_column"]], truncation=True, padding=True, max_length=512)
        
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        train_dataset = train_dataset.rename_column(model_config["label_column"], "labels")
        train_dataset = train_dataset.remove_columns([model_config["text_column"]])

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args={"output_dir": model_config["output_model_dir"], "num_train_epochs": model_config["num_epochs"]}
        )

        trainer.train()
        trainer.save_model(model_config["output_model_dir"])
        tokenizer.save_pretrained(model_config["output_model_dir"])

        wandb.finish()
        print(f"🚀 Training for {model_name} completed!")

    background_tasks.add_task(train)
    return {"message": f"Training for {model_name} started."}

# Request Schema for Evaluation
class EvalRequest(BaseModel):
    model_name: str
    test_file: str

@app.post("/evaluate/")
def evaluate_model(request: EvalRequest):
    """Evaluate a trained model on a test dataset"""
    try:
        model, tokenizer = load_model(request.model_name)
        df = pd.read_csv(request.test_file, sep="\t")
        dataset = Dataset.from_pandas(df)

        def preprocess_function(examples):
            return tokenizer(examples["Sentence"], truncation=True, padding=True, max_length=512)

        dataset = dataset.map(preprocess_function, batched=True)
        dataset = dataset.rename_column("Label", "labels")
        dataset = dataset.remove_columns(["Sentence"])

        trainer = Trainer(model=model)
        predictions = trainer.predict(dataset)
        logits, labels = predictions.predictions, predictions.label_ids
        metrics = compute_generalization_metrics(logits, labels)

        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

