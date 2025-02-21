import optuna
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_embeddings(texts, tokenizer, model, batch_size=8):  # Reduce batch size
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

def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    return tokenizer, model

def fit_gmm_incrementally(data_path, tokenizer, model, chunk_size=5000, batch_size=8, 
                          n_components=2, covariance_type='full', tol=1e-3, max_iter=100):
    temp_embeddings = []
    reader = pd.read_csv(data_path, sep='\t', on_bad_lines='skip', chunksize=chunk_size)

    for chunk in reader:
        chunk = chunk.dropna(subset=['Sentence'])
        if chunk.empty:
            continue

        embeddings = generate_embeddings(chunk['Sentence'].tolist(), tokenizer, model, batch_size)
        temp_embeddings.append(embeddings)
        
        del embeddings
        torch.cuda.empty_cache()

    if temp_embeddings:
        all_embeddings = np.vstack(temp_embeddings)
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                              tol=tol, max_iter=max_iter, reg_covar=1e-3, random_state=42)
        gmm.fit(all_embeddings)
        return gmm
    else:
        return None

def objective(trial, data_path, output_dir, tokenizer, model):
    n_components = trial.suggest_int('n_components', 2, 10)
    covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
    tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
    max_iter = trial.suggest_int('max_iter', 50, 300)

    gmm = fit_gmm_incrementally(data_path, tokenizer, model, n_components=n_components, 
                                covariance_type=covariance_type, tol=tol, max_iter=max_iter)

    return gmm.bic(np.vstack(temp_embeddings)) if gmm else float("inf")

def optimize_gmm(data_path, output_dir, n_trials=30):
    tokenizer, model = load_bert()
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(lambda trial: objective(trial, data_path, output_dir, tokenizer, model), 
                   n_trials=n_trials, n_jobs=1) 
    return study

# Run Optimization
data_path = "./combined_data/Sent-Spacy/combined_data_en.ol"
study = optimize_gmm(data_path, output_dir="./Sent-en-GMM-Bert", n_trials=30)
