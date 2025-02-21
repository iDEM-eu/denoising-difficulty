import os
import pandas as pd
from datetime import datetime

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss, brier_score_loss
import wandb
import torch
import re
from scipy.special import softmax
from scipy.stats import entropy
import numpy as np


text_column = "Clean_Text"  # name of the text column
label_column = "Label"  # name of the label column
seed=42

def clean_text(text):
    """Remove HTML tags and URLs from text."""
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.MULTILINE)
    return text


def prepare_datasets(data_file, sep=",", test_size=0):
    """
    Load the dataset and perform a random split.
    Args:
        data_file (str): Path to the CSV file containing data.
        test_size (float): Proportion of the data to include in the test split.
    Returns:
        train_dataset, val_dataset: Hugging Face Datasets for training and validation.
    """
    df = pd.read_csv(data_file)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns {text_column} and/or {label_column} not found in the dataset.")

    df[text_column] = df[text_column].apply(clean_text)

    if test_size>0:
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
    else:
        train_dataset = Dataset.from_pandas(df)
        val_dataset=None

    return train_dataset, val_dataset

# Function to calculate generalization metrics
def compute_generalization_metrics(logits, labels):
    """
    Compute generalization metrics including log loss, calibration error, and entropy.
    Args:
        logits (np.ndarray): Model logits or probabilities.
        labels (np.ndarray): True labels.
    Returns:
        dict: Generalization metrics.
    """
    # Convert logits to probabilities if needed
    probabilities = softmax(logits, axis=1)

    # Calculate predicted labels
    preds = probabilities.argmax(axis=1)

    # Standard metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    roc_auc = roc_auc_score(labels, probabilities[:, 1]) if probabilities.shape[1] > 1 else None

    # Generalization metrics
    log_loss_value = log_loss(labels, probabilities)
    brier_score = brier_score_loss(labels, probabilities[:, 1])
    prediction_entropy = np.mean(entropy(probabilities, axis=1))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Log_Loss": log_loss_value,
        "Brier_Score": brier_score,
        "Prediction_Entropy": prediction_entropy,
    }

def compute_metrics(eval_pred, trainer=None, train_dataset=None):
    """
    Compute metrics including generalization gap if trainer and train_dataset are provided.
    Args:
        eval_pred: Tuple of (logits, labels).
        trainer: Hugging Face Trainer instance (optional).
        train_dataset: Training dataset (optional, for generalization gap).
    Returns:
        dict: Evaluation metrics and generalization gaps (if applicable).
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    # Standard metrics
    metrics = compute_generalization_metrics(logits, labels)

    # Compute generalization gap if trainer and train_dataset are provided
    if trainer and train_dataset:
        train_results = trainer.evaluate(train_dataset)
        if "eval_accuracy" in train_results:
            metrics["accuracy_gap"] = train_results["eval_accuracy"] - metrics["accuracy"]
        if "eval_loss" in train_results:
            metrics["loss_gap"] = train_results["eval_loss"] - metrics["log_loss"]

    return metrics


def train_model(
    model_name,
    train_dataset,
    val_dataset,
    output_dir,
    learning_rate=1e-5,
    epochs=3,
    batch_size=8,
    eval_steps=500,
    project_name='Complexity'
):
    """
    Train the model using Hugging Face Trainer.
    Args:
        model_name (str): Model name from Hugging Face.
        train_dataset, val_dataset: Hugging Face Datasets for training and validation.
        output_dir (str): Directory to save the model and tokenizer.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        eval_steps (int): Number of steps between evaluations.
        project_name (str): the name for reporting to Wandb.
    """
    if project_name:
        wandb.init(project=project_name, name=model_name, reinit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    def preprocess_function(examples):
        return tokenizer(
            examples[text_column], truncation=True, padding=True, max_length=512
        )

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.rename_column("Label", "labels")

    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_val = tokenized_val.rename_column("Label", "labels")

    tokenized_train = tokenized_train.remove_columns(["Clean_Text"])
    tokenized_val = tokenized_val.remove_columns(["Clean_Text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")

# **Compute Generalization Gaps**
    train_results = trainer.evaluate(tokenized_train)
    val_results = trainer.evaluate(tokenized_val)

    generalization_gaps = {
        metric: train_results[f"eval_{metric}"] - val_results[f"eval_{metric}"]
        for metric in ["accuracy", "loss", "f1"]
        if f"eval_{metric}" in train_results and f"eval_{metric}" in val_results
    }

    print(f"Generalization Gaps: {generalization_gaps}")
    if project_name:
        wandb.log(generalization_gaps)

    # Return metrics for reference
    return generalization_gaps

def test_model(model_name, test_dataset):
    """
    Evaluate the model on a specific test dataset and compute generalization metrics.
    Args:
        model_name: Pretrained model for sequence classification.
        test_dataset: Hugging Face Dataset to evaluate.
        text_column (str): Name of the text column in the dataset.
        label_column (str): Name of the label column in the dataset.
    Returns:
        dict: Computed metrics.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding=True, max_length=512)

    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    tokenized_test = tokenized_test.rename_column(label_column, "labels")
    tokenized_test = tokenized_test.remove_columns([text_column])

    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_test)

    # Extract logits and labels
    logits, labels = predictions.predictions, predictions.label_ids

    # Compute metrics using compute_generalization_metrics
    metrics = compute_generalization_metrics(logits, labels)
    return metrics


