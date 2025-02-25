import os
import json
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import wandb
from utils.metrics_utils import compute_generalization_metrics, compute_metrics
from transformers import EarlyStoppingCallback

# Load Configurations
def load_config(config_path, model_name):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["models"][model_name]

# Initialize WandB
def initialize_wandb(project_name, run_name):
    wandb.init(project=project_name, name=run_name)

# Label Smoothing
def label_smoothing(labels, num_classes, epsilon):
    batch_size = labels.size(0)
    smooth_labels = torch.full((batch_size, num_classes), epsilon / num_classes).to(labels.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon)
    return smooth_labels

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, smoothing_factor=0.1):
        self.encodings = tokenizer(
            list(df['Sentence'].values),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        self.labels = torch.tensor(df['Label'].values, dtype=torch.long)
        self.smoothing_factor = smoothing_factor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        if self.smoothing_factor > 0:
            item["labels"] = label_smoothing(
                item["labels"].unsqueeze(0), num_classes=2, epsilon=self.smoothing_factor
            ).squeeze(0)
        return item

# Training Function
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_wandb(config["wandb_project"], config["wandb_run"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    df = pd.read_csv(config["dataset_path"], sep=config["separator"])
    train_df, val_df = train_test_split(df, test_size=config["valsplit"], random_state=config["seed"])
    train_dataset = TextDataset(train_df, tokenizer, smoothing_factor=config["smoothing_factor"])
    val_dataset = TextDataset(val_df, tokenizer, smoothing_factor=config["smoothing_factor"])
    model = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir=config["output_model_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        logging_dir="./logs",
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
)

    trainer.train()
    trainer.save_model(config["output_model_dir"])
    tokenizer.save_pretrained(config["output_model_dir"])
    wandb.finish()

# Testing Function
def test_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["output_model_dir"])
    test_df = pd.read_csv(config["testfile"], sep=config["separator"])
    test_dataset = TextDataset(test_df, tokenizer, smoothing_factor=config["smoothing_factor"])
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = test_dataset.labels.numpy()
    metrics = compute_generalization_metrics(logits, labels)
    print("Test Metrics:", metrics)

# Main Function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Model name from config")
    args = parser.parse_args()
    config = load_config(args.config, args.model)
    train_model(config)
    
