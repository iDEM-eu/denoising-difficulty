import os
import json
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
import wandb
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

def load_config(config_path, model_name):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["models"][model_name]

def initialize_wandb(project_name, run_name):
    wandb.init(project=project_name, name=run_name)

def clean_text(text):
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.MULTILINE)
    return text

def prepare_datasets(config):
    df = pd.read_csv(config["dataset_path"], sep=config["separator"])
    df[config["text_column"]] = df[config["text_column"]].apply(clean_text)
    train_df, val_df = train_test_split(df, test_size=config["valsplit"], random_state=config["seed"])
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_wandb(config["wandb_project"], config["wandb_run"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    train_dataset, val_dataset = prepare_datasets(config)
    model = AutoModelForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=config["num_classes"]).to(device)
    
    def preprocess_function(examples):
        return tokenizer(examples[config["text_column"]], truncation=True, padding=True, max_length=512)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True).rename_column(config["label_column"], "labels")
    val_dataset = val_dataset.map(preprocess_function, batched=True).rename_column(config["label_column"], "labels")
    
    train_dataset = train_dataset.remove_columns([config["text_column"]])
    val_dataset = val_dataset.remove_columns([config["text_column"]])
    
    training_args = TrainingArguments(
        output_dir=config["output_model_dir"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_dir=f"{config['output_model_dir']}/logs",
        save_total_limit=2,
        save_strategy="epoch",
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(config["output_model_dir"])
    tokenizer.save_pretrained(config["output_model_dir"])
    wandb.finish()

def test_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["output_model_dir"])
    test_df = pd.read_csv(config["testfile"], sep=config["separator"])
    test_dataset = Dataset.from_pandas(test_df)
    
    def preprocess_function(examples):
        return tokenizer(examples[config["text_column"]], truncation=True, padding=True, max_length=512)
    
    test_dataset = test_dataset.map(preprocess_function, batched=True).rename_column(config["label_column"], "labels")
    test_dataset = test_dataset.remove_columns([config["text_column"]])
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    logits, labels = predictions.predictions, predictions.label_ids
    metrics = compute_generalization_metrics(logits, labels)
    print("Test Metrics:", metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Model name from config")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    args = parser.parse_args()
    config = load_config(args.config, args.model)
    
    if args.mode == "train":
        train_model(config)
    elif args.mode == "test":
        test_model(config)
