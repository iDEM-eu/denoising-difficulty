import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import wandb
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def initialize_wandb(project_name, run_name):
    wandb.init(project=project_name, name=run_name)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, model_name, device, batch_size=32):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embeddings = self.generate_embeddings()

    def generate_embeddings(self):
        all_embeddings = []
        self.model.to(self.device)
        for i in range(0, len(self.sentences), self.batch_size):
            batch_texts = self.sentences[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            torch.cuda.empty_cache()
        return np.vstack(all_embeddings)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32).to(self.device),
            torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)
        )

class AdvancedNN(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for sentences, labels in dataloader:
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences).squeeze()
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for sentences, labels in dataloader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences).squeeze()
            all_logits.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    metrics = compute_generalization_metrics(np.array(all_logits).reshape(-1, 1), np.array(all_labels))
    wandb.log(metrics)
    return metrics

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)

def main(config_path):
    config = load_config(config_path)
    device = get_device()
    initialize_wandb(config["wandb_project"], config["wandb_run"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"])
    model = AdvancedNN(config["model_input_dim"])
    criterion = getattr(nn, config["loss_function"])()
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=config["learning_rate"])
    df = pd.read_csv(config["dataset_path"], sep=config["separator"], on_bad_lines='skip')
    dataset = TextDataset(df[config["text_column"]].tolist(), df[config["label_column"]].tolist(), tokenizer, config["tokenizer_model"], device)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    train_model(model, dataloader, optimizer, criterion, num_epochs=config["num_epochs"], device=device)
    metrics = evaluate_model(model, dataloader, device)
    print("Evaluation Metrics:", metrics)
    save_model(model, tokenizer, config["output_model_dir"])
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()
    main(args.config)
