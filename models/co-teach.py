import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AdamW
)
from sklearn.model_selection import train_test_split
import wandb
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

def load_config(config_path, model_name):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["models"][model_name]

def initialize_wandb(project_name, run_name):
    wandb.init(project=project_name, name=run_name)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, text_column, label_column):
        self.df = df.reset_index(drop=True)
        self.texts = self.df[text_column].tolist()
        self.labels = self.df[label_column].tolist()
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def prepare_datasets(config, tokenizer):
    df = pd.read_csv(config["dataset_path"], sep=config["separator"])
    train_df, val_df = train_test_split(df, test_size=config["valsplit"], random_state=config["seed"])
    train_dataset = TextDataset(train_df, tokenizer, config["text_column"], config["label_column"])
    val_dataset = TextDataset(val_df, tokenizer, config["text_column"], config["label_column"])
    return train_dataset, val_dataset

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_wandb(config["wandb_project"], config["wandb_run"])

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    model1 = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=2).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=2).to(device)

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
        metric_for_best_model="f1",
        greater_is_better=True,
        early_stopping_patience=config["early_stopping_patience"]
    )

    trainer1 = Trainer(
        model=model1,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer2 = Trainer(
        model=model2,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train both models
    trainer1.train()
    trainer2.train()

    # Evaluate and pick the best model
    val_metrics1 = trainer1.evaluate(val_dataset)
    val_metrics2 = trainer2.evaluate(val_dataset)

    best_trainer = trainer1 if val_metrics1["eval_f1"] > val_metrics2["eval_f1"] else trainer2
    best_model_path = os.path.join(config["output_model_dir"], "best_model")

    # Save the best model
    best_trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"🚀 Best model saved at {best_model_path}")

    wandb.finish()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend(labels.cpu().numpy())
    return compute_generalization_metrics(np.array(all_logits), np.array(all_labels)), all_logits, all_labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Model name from config")
    parser.add_argument("--mode", type=str, choices=["train"], required=True, help="Mode: train")
    args = parser.parse_args()
    config = load_config(args.config, args.model)
    
    if args.mode == "train":
        train_model(config)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_wandb(config["wandb_project"], config["wandb_run"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    model1 = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=2).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=2).to(device)

    optimizer1 = AdamW(model1.parameters(), lr=config["learning_rate"])
    optimizer2 = AdamW(model2.parameters(), lr=config["learning_rate"])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config["batch_size"])
    val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=config["batch_size"])

    best_model_path = os.path.join(config["output_model_dir"], "best_model")
    os.makedirs(config["output_model_dir"], exist_ok=True)

    best_f1 = 0.0  # Track the best model

    for epoch in range(1, config["num_epochs"] + 1):
        model1.train()
        model2.train()

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            out1 = model1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out2 = model2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss1 = out1.loss
            loss2 = out2.loss

            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

        # Evaluate both models on validation set
        val_metrics1, _, _ = evaluate_model(model1, val_dataloader, device)
        val_metrics2, _, _ = evaluate_model(model2, val_dataloader, device)

        print(f"Epoch {epoch}: Model1 F1={val_metrics1['f1']:.4f}, Model2 F1={val_metrics2['f1']:.4f}")

        # Determine the best model
        if val_metrics1["f1"] > best_f1:
            best_f1 = val_metrics1["f1"]
            best_model = model1
        if val_metrics2["f1"] > best_f1:
            best_f1 = val_metrics2["f1"]
            best_model = model2

    # Save the best model using Hugging Face methods
    best_model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"🚀 Best model saved at {best_model_path}")

    wandb.finish()

