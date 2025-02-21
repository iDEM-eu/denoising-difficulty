import optuna
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.mixture import GaussianMixture
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_embeddings(texts, tokenizer, model, batch_size=8):
    all_embeddings = []
    model.to(device)
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

        inputs = {
            "input_ids": inputs["input_ids"].to(device).to(torch.long),
            "attention_mask": inputs["attention_mask"].to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].to("cpu", dtype=torch.float32).numpy()
        all_embeddings.append(embeddings)
        
        del inputs, outputs
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)

def load_sbert():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    return tokenizer, model

def optimize_gmm(data_path, output_dir, n_trials=30):
    tokenizer, model = load_sbert()
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, data_path, output_dir, tokenizer, model), 
                   n_trials=n_trials, n_jobs=1)  # Reduce parallel jobs

    return study

# Run Optimization
data_path = "./combined_data/Sent-Spacy/combined_data_en.ol"
study = optimize_gmm(data_path, output_dir="./Sent-en-GMM-SBERT", n_trials=30)
