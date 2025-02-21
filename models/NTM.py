import os
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
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
        self.encodings = tokenizer(
            list(df[text_column].values),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def train_and_validate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_wandb(config["wandb_project"], config["wandb_run"])
    tokenizer = BertTokenizer.from_pretrained(config["tokenizer_model"])
    df = pd.read_csv(config["dataset_path"], sep=config["separator"])
    train_df, val_df = train_test_split(df, test_size=config["valsplit"], random_state=config["seed"])
    train_dataset = TextDataset(train_df, tokenizer, config["text_column"], config["label_column"])
    val_dataset = TextDataset(val_df, tokenizer, config["text_column"], config["label_column"])
    
    # Noise Transition Matrix
    conf_matrix = pd.crosstab(train_df[config["label_column"]], train_df["Noisy_Label"], rownames=['True'], colnames=['Noisy'], normalize='index')
    T = torch.tensor(conf_matrix.values, dtype=torch.float32)
    epsilon = config["epsilon"]
    T = T * (1 - epsilon) + epsilon / T.size(1)
    T_inv = torch.inverse(T)
    
    model = BertForSequenceClassification.from_pretrained(config["tokenizer_model"], num_labels=len(df[config["label_column"].unique()])).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    
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
        early_stopping_patience=config["early_stopping_patience"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(config["output_model_dir"])
    tokenizer.save_pretrained(config["output_model_dir"])
    
    # Validation Step
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    labels = val_dataset.labels.numpy()
    metrics = compute_generalization_metrics(logits, labels)
    print("Validation Metrics:", metrics)
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--model", type=str, required=True, help="Model name from config")
    args = parser.parse_args()
    config = load_config(args.config, args.model)
    train_and_validate(config)
