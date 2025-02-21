#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss, brier_score_loss
from torch.utils.data import Dataset
import wandb
from sklearn.model_selection import train_test_split
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

def label_smoothing(labels, num_classes, epsilon):
    batch_size = labels.size(0)
    smooth_labels = torch.full((batch_size, num_classes), epsilon / num_classes).to(labels.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon)
    return smooth_labels

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, smoothing_factor=0.1):
        self.encodings = tokenizer(list(df['Clean_Text'].values), padding=True, truncation=True, max_length=512, return_tensors='pt')
        self.labels = torch.tensor(df['Label'].values, dtype=torch.long)
        self.smoothing_factor = smoothing_factor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]

        if self.smoothing_factor > 0:
            item["labels"] = label_smoothing(item["labels"].unsqueeze(0), num_classes=2, epsilon=self.smoothing_factor).squeeze(0)
        return item

def train_model(args):
    if args.use_wandb:
        wandb.init(project=args.projectname, name="Label_Smoothing_Transformer")

   
    tokenizer = AutoTokenizer.from_pretrained(args.mname)
    
    df = pd.read_csv(args.inputfile, sep=args.sep)
    train_df, val_df = train_test_split(df, test_size=args.valsplit, random_state=args.seed)
    train_dataset = TextDataset(train_df, tokenizer, smoothing_factor=args.smoothing_factor)
    val_dataset = TextDataset(val_df, tokenizer, smoothing_factor=args.smoothing_factor)

    # Use AutoModelForSequenceClassification to dynamically load models
    model = AutoModelForSequenceClassification.from_pretrained(args.mname, num_labels=2)

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
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        early_stopping_patience=args.early_stopping
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

    if args.use_wandb:
        wandb.finish()

def test_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.mname)
    model = AutoModelForSequenceClassification.from_pretrained(args.local)

    test_df = pd.read_csv(args.testfile, sep=args.sep)
    test_dataset = TextDataset(test_df, tokenizer, smoothing_factor=args.smoothing_factor)
    
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = test_dataset.labels.numpy()

    print(args.testfile)
    print("\t".join([f"{k} {logits[k]:.3f}" for k in logits]))
