import os
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss, brier_score_loss
import torch
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import argparse
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Model Testing API")

class TestRequest(BaseModel):
    model_path: str
    test_config: str

def clean_text(text):
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text, flags=re.MULTILINE)
    return text

def prepare_test_dataset(data_file, text_column, label_column):
    df = pd.read_csv(data_file, sep='\t', on_bad_lines='skip')
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns {text_column} and/or {label_column} not found in the dataset.")
    df[text_column] = df[text_column].apply(clean_text)
    return Dataset.from_pandas(df)

def compute_generalization_metrics(logits, labels):
    probabilities = softmax(logits, axis=1)
    preds = probabilities.argmax(axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    roc_auc = roc_auc_score(labels, probabilities[:, 1]) if probabilities.shape[1] > 1 else None
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Log_Loss": log_loss(labels, probabilities),
        "Brier_Score": brier_score_loss(labels, probabilities[:, 1]),
        "Prediction_Entropy": np.mean(entropy(probabilities, axis=1)),
    }

@app.post("/test")
def evaluate_model(request: TestRequest):
    with open(request.test_config, "r") as f:
        config = json.load(f)
    test_files = config["test_files"]
    text_column = config["text_column"]
    label_column = config["label_column"]
    
    model = AutoModelForSequenceClassification.from_pretrained(request.model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(request.model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    evaluation_results = []
    for lang, test_file in test_files.items():
        print(f"Evaluating {lang} dataset...")
        test_dataset = prepare_test_dataset(test_file, text_column, label_column)
        def preprocess_function(examples):
            return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=512)
        tokenized_test = test_dataset.map(preprocess_function, batched=True)
        tokenized_test = tokenized_test.rename_column(label_column, "labels")
        tokenized_test = tokenized_test.remove_columns([text_column])
        trainer = Trainer(model=model, eval_dataset=tokenized_test, data_collator=DataCollatorWithPadding(tokenizer), args=TrainingArguments(output_dir="./results"))
        predictions = trainer.predict(tokenized_test)
        metrics = compute_generalization_metrics(predictions.predictions, predictions.label_ids)
        evaluation_results.append({"Language": lang, **metrics})
    evaluation_df = pd.DataFrame(evaluation_results)
    output_filename = f"Sent-Evaluation_{request.model_path.split('/')[-1]}.csv"
    evaluation_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
    return {"message": "Evaluation completed", "results": evaluation_results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test datasets")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_config", type=str, required=True, help="Path to the test files configuration JSON")
    args = parser.parse_args()
    
    evaluate_model(TestRequest(model_path=args.model, test_config=args.test_config))
