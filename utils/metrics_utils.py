import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss, brier_score_loss

def compute_generalization_metrics(logits, labels):
    """
    Compute generalization metrics including log loss, calibration error, and entropy.
    Args:
        logits (np.ndarray): Model logits or probabilities.
        labels (np.ndarray): True labels.
    Returns:
        dict: Generalization metrics.
    """
    # Convert logits to probabilities
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
            metrics["accuracy_gap"] = train_results["eval_accuracy"] - metrics["Accuracy"]
        if "eval_loss" in train_results:
            metrics["loss_gap"] = train_results["eval_loss"] - metrics["Log_Loss"]
    
    return metrics
