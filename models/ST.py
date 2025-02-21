import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, brier_score_loss, roc_auc_score, confusion_matrix
from scipy.special import softmax
from scipy.stats import entropy
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

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
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for sentences, labels in dataloader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences).squeeze()
            all_logits.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    probabilities = softmax(all_logits.reshape(-1, 1), axis=1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    log_loss_value = log_loss(all_labels, probabilities)
    brier_score = brier_score_loss(all_labels, probabilities[:, 0])
    prediction_entropy = np.mean(entropy(probabilities, axis=1))
    conf_matrix = confusion_matrix(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, probabilities)
    except ValueError:
        roc_auc = None
    
    wandb.log({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC AUC": roc_auc,
        "Log Loss": log_loss_value,
        "Brier Score": brier_score,
        "Prediction Entropy": prediction_entropy
    })
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)

def main():
    device = get_device()
    initialize_wandb("General-model-English", "Sent-ST")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AdvancedNN(768)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    df = pd.read_csv("./combined_data/Sent-Spacy/combined_data_en.ol", sep='\t', on_bad_lines='skip')
    dataset = TextDataset(df['Sentence'].tolist(), df['Label'].tolist(), tokenizer, "bert-base-multilingual-cased", device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_model(model, dataloader, optimizer, criterion, num_epochs=5, device=device)
    save_model(model, tokenizer, "./Sent-ST-bert-model")
    wandb.finish()

if __name__ == "__main__":
    main()
