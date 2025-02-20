#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from scipy.special import softmax
from scipy.stats import entropy
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss, brier_score_loss
from torch.utils.data import Dataset
import wandb
from sklearn.model_selection import train_test_split

def dynamic_label_smoothing(epoch, total_epochs, initial_epsilon):
    return initial_epsilon * (1 - (epoch / total_epochs))

def adaptive_probability_threshold(epoch, total_epochs, min_threshold, max_threshold):
    return min_threshold + (max_threshold - min_threshold) * (epoch / total_epochs)

def label_smoothing(labels, num_classes, epsilon):
    batch_size = labels.size(0)
    smooth_labels = torch.full((batch_size, num_classes), epsilon / num_classes).to(labels.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon)
    return smooth_labels

parser = argparse.ArgumentParser(description="A Transformer Model with Label Smoothing")
parser.add_argument('-m', '--mname', type=str, default='bert-base-multilingual-cased', help='Model name from HuggingFace')
parser.add_argument('-l', '--local', type=str, default=None, help='Directory for saving the local model')
parser.add_argument('-p', '--projectname', type=str, default=None, help='Project name for WandB')
parser.add_argument('--use_wandb', action='store_true', help='Flag to enable WandB logging')
parser.add_argument('-i', '--inputfile', type=str, default=None, help='Training dataset file')
parser.add_argument('-t', '--testfile', type=str, default=None, help='Test dataset file')
parser.add_argument('--sep', type=str, default='\t', help='Field separator')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--eval_steps', type=int, default=1000)
parser.add_argument('--valsplit', type=float, default=0.2)
parser.add_argument('-v', '--verbosity', type=int, default=1)
args = parser.parse_args()

if args.use_wandb:
    wandb.init(project=args.projectname, name="Label_Smoothing_Transformer")

tokenizer = BertTokenizer.from_pretrained(args.mname)

def prepare_datasets(file_path, sep, valsplit=None):
    df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
    label_mapping = {"wiki": 1, "vikidia": 0, 1: 1, 0: 0}
    df['Label'] = df['Label'].map(label_mapping).fillna(-1).astype(int)
    
    if valsplit:
        train_df, val_df = train_test_split(df, test_size=valsplit, random_state=args.seed)
        return train_df, val_df
    return df, None

if args.inputfile:
    train_df, val_df = prepare_datasets(args.inputfile, args.sep, args.valsplit)
    
    train_dataset = TextDataset(train_df, tokenizer, initial_smoothing=0.1)
    val_dataset = TextDataset(val_df, tokenizer, initial_smoothing=0.1)
    
    model = BertForSequenceClassification.from_pretrained(args.mname, num_labels=2)
    training_args = TrainingArguments(
        output_dir=args.local,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        report_to="wandb" if args.use_wandb else "none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.local)
    tokenizer.save_pretrained(args.local)

if args.testfile:
    test_df, _ = prepare_datasets(args.testfile, args.sep)
    test_dataset = TextDataset(test_df, tokenizer, initial_smoothing=0.1)
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = test_dataset.labels.numpy()
    
    metrics = compute_generalization_metrics(logits, labels, threshold=0.5)
    print(args.testfile)
    print("\t".join([f"{k} {metrics[k]:.3f}" for k in metrics]))

if args.use_wandb:
    wandb.finish()
