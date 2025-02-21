import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, log_loss, brier_score_loss
)
from scipy.special import softmax
from scipy.stats import entropy

import wandb
from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW
)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
# ----------------------------------------------------------------------
# 1) Hyperparameters & Paths
# ----------------------------------------------------------------------
OUTPUT_DIR = "./Sent-en-coteaching-output"
DATA_FILE = "./combined_data/Sent-Spacy/combined_data_en.ol"
TEST_FILES = {
    "fr": "./combined_data/Sent-Spacy/combined_data_fr.ol",
    "es": "./combined_data/Sent-Spacy/combined_data_es.ol",
    "it": "./combined_data/Sent-Spacy/combined_data_it.ol",
    "ru": "./combined_data/Sent-Spacy/combined_data_ru.ol",
    "ca": "./combined_data/Sent-Spacy/combined_data_ca.ol",
}
LABEL_MAPPING = {"wiki": 1, "vikidia": 0, 1: 1, 0: 0}
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5

# **Linearly scheduled forget rate** from RHO_MIN at epoch=1 to RHO_MAX at epoch=N
RHO_MIN = 0.0   
RHO_MAX = 0.3   

CONFIDENCE_THRESHOLD = 0.6  # For optional “noisy sample” detection after training

# ----------------------------------------------------------------------
# 2) Read & Prepare Data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE, sep='\t', on_bad_lines='skip')
df["Label"] = df["Label"].map(LABEL_MAPPING)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

class TextDataset(Dataset):
    """
    A dataset that stores the *entire* DataFrame internally as self.df.
    We store each row's encodings + text + label + the *local index*.

    That way, we can later retrieve row i with dataset.dataset.df.iloc[idx]
    if we want the original "Sentence" or other columns.
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.texts = self.df["Sentence"].tolist()
        self.labels = self.df["Label"].tolist()
        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids":        self.encodings["input_ids"][idx],
            "attention_mask":   self.encodings["attention_mask"][idx],
            "label":            torch.tensor(self.labels[idx], dtype=torch.long),
            "text":             self.texts[idx],
            # 'idx' is the local index *within this dataset* (not necessarily the original DF row if we've done a split)
            "idx":              idx
        }

full_dataset = TextDataset(df)

# Train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_dataloader   = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 # ----------------------------------------------------------------------
# 3) Initialize Models & Optimizers
# ----------------------------------------------------------------------
model1 = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2).to(device)
model2 = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2).to(device)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
# Use XLM-RoBERTa instead of BERT
#model1 = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2).to(device)
#model2 = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2).to(device)


optimizer1 = AdamW(model1.parameters(), lr=LR)
optimizer2 = AdamW(model2.parameters(), lr=LR)

# ----------------------------------------------------------------------
# 4) Metrics & Evaluation
# ----------------------------------------------------------------------
def compute_generalization_metrics(logits, labels):
    probs = softmax(logits, axis=1)
    preds = probs.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = None
    ll = log_loss(labels, probs)
    brier = brier_score_loss(labels, probs[:, 1])
    pred_entropy = np.mean(entropy(probs, axis=1))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": ll,
        "brier_score": brier,
        "prediction_entropy": pred_entropy,
    }

def evaluate_model(model, dataloader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()

            all_logits.extend(logits)
            all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    metrics = compute_generalization_metrics(all_logits, all_labels)
    return metrics, all_logits, all_labels

# ----------------------------------------------------------------------
# 5) Advanced Co-Teaching Loop with Linear Forget Rate
# ----------------------------------------------------------------------
def linear_forget_rate(epoch, total_epochs, rho_min=0.0, rho_max=0.3):
    """
    Linearly grow forget rate from rho_min at epoch=1 up to rho_max at epoch=total_epochs.
    """
    if total_epochs <= 1:
        return rho_max  # fallback if only 1 epoch
    return rho_min + (rho_max - rho_min) * (epoch - 1) / (total_epochs - 1)

def train_co_teaching(
    model1, model2,
    train_dl, val_dl,
    optimizer1, optimizer2,
    train_dataset,  
    epochs=5,
    rho_min=0.0,
    rho_max=0.3,
    early_stopping_patience=2,  
    checkpoint_interval=2  # Save a checkpoint every N epochs
):
    best_f1 = 0.0  
    best_model_path = os.path.join(OUTPUT_DIR, "best_overall_model.pth")
    best_epoch = 0
    epochs_without_improvement = 0  
    best_training_indices = None  

    for epoch in range(1, epochs + 1):
        model1.train()
        model2.train()

        forget_rate = linear_forget_rate(epoch, epochs, rho_min, rho_max)
        print(f"\n=== Epoch {epoch}/{epochs} ===, forget_rate={forget_rate:.3f}")

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            original_idxs = batch["idx"]

            # Forward pass for both models
            out1 = model1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out2 = model2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits1, logits2 = out1.logits, out2.logits
            loss1_all = F.cross_entropy(logits1, labels, reduction='none')
            loss2_all = F.cross_entropy(logits2, labels, reduction='none')

            idxs_sorted_by_model1 = torch.argsort(loss1_all)
            idxs_sorted_by_model2 = torch.argsort(loss2_all)

            batch_size = labels.size(0)
            num_keep = int((1.0 - forget_rate) * batch_size)

            idx_keep_for_model1 = idxs_sorted_by_model2[:num_keep]
            idx_keep_for_model2 = idxs_sorted_by_model1[:num_keep]

            loss1_clean = loss1_all[idx_keep_for_model1]
            loss2_clean = loss2_all[idx_keep_for_model2]

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss1_mean = loss1_clean.mean() if len(loss1_clean) > 0 else loss1_all.mean()
            loss2_mean = loss2_clean.mean() if len(loss2_clean) > 0 else loss2_all.mean()

            loss1_mean.backward()
            loss2_mean.backward()

            optimizer1.step()
            optimizer2.step()

        # Validation Evaluation
        metrics1, _, _ = evaluate_model(model1, val_dl)
        metrics2, _, _ = evaluate_model(model2, val_dl)

        print(f" Model1 val metrics: {metrics1}")
        print(f" Model2 val metrics: {metrics2}")

        # Determine best model this epoch
        best_f1_model = "Model1" if metrics1["f1"] > metrics2["f1"] else "Model2"
        best_f1_this_epoch = max(metrics1["f1"], metrics2["f1"])

        if best_f1_this_epoch > best_f1:
            best_f1 = best_f1_this_epoch
            best_epoch = epoch
            epochs_without_improvement = 0  

            # Save the best model
            best_model = model1 if best_f1_model == "Model1" else model2
            torch.save(best_model.state_dict(), best_model_path)
            print(f"🚀 New Best Model ({best_f1_model}) saved with F1={best_f1:.4f} at Epoch {epoch}")

            best_training_indices = [train_dataset.indices[idx] for idx in range(len(train_dataset))]
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

        # Save periodic checkpoint
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model1_state_dict': model1.state_dict(),
                'model2_state_dict': model2.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'best_f1': best_f1
            }, checkpoint_path)
            print(f"💾 Checkpoint saved at {checkpoint_path}")

        # Early stopping condition
        if epochs_without_improvement >= early_stopping_patience:
            print(f"⏹️ Early stopping triggered at Epoch {epoch}. Best model was from Epoch {best_epoch}.")
            break

    print("🏁 Co-teaching training finished.")

    # Save the best training data
    if best_training_indices is not None:
        best_training_data = train_dataset.dataset.df.iloc[best_training_indices]
        best_training_data.to_csv(os.path.join(OUTPUT_DIR, f"best_training_data_epoch_{best_epoch}.csv"), index=False)
        print(f"📂 Best training data from Epoch {best_epoch} saved.")

    return best_model_path, best_epoch

# ----------------------------------------------------------------------
# 6) Writing Discarded Samples
# ----------------------------------------------------------------------
def save_discarded_samples(discarded_indices, dataset, output_path, label="Model"):
    """
    - `discarded_indices` is a set of local indices from the *random_split* dataset.
    - We need to access the dataset.dataset to get the *underlying* dataset.
    - Then we do dataset.dataset.df.iloc[...] to get the original row from the DataFrame.
    """
    if not discarded_indices:
        print(f"[{label}] No samples ever discarded.")
        return

    # The random_split dataset has a reference: dataset.dataset => full_dataset
    # So full_dataset.df is the original DataFrame. But we have to confirm the mapping:
    #   random_split gives you a Subset. If we want row i, we look at subset.indices[i].
    #   Then inside full_dataset, that's the row.
    # We'll do something like: subset_idx -> global_idx -> original df row.

    # The catch: random_split has .indices, which is the list of *global* indices in full_dataset.
    # So for each idx in `discarded_indices`, we find the global index in .indices,
    # then retrieve that row from full_dataset.df.
    if not hasattr(dataset, 'indices'):
        # If we used random_split, 'dataset' is a Subset with an .indices attribute
        print(f"[{label}] The dataset is not a Subset with .indices. Skipping.")
        return

    subset_indices = dataset.indices  # list of global indices in the underlying full_dataset
    full_df = dataset.dataset.df

    rows = []
    for local_idx in discarded_indices:
        local_idx = int(local_idx) 
        # local_idx is the position *within the subset*
        if local_idx < 0 or local_idx >= len(subset_indices):
            continue
        global_idx = subset_indices[local_idx]
        row_data = full_df.iloc[global_idx]

        rows.append({
            "Global_Index": global_idx,
            "Local_Index_in_Subset": local_idx,
            "Sentence": row_data["Sentence"],
            "True_Label": row_data["Label"]
        })

    if not rows:
        print(f"[{label}] All discarded indices were out of range; or none found.")
        return

    df_disc = pd.DataFrame(rows)
    df_disc.to_csv(output_path, index=False)
    print(f"[{label}] Discarded samples saved to {output_path} (n={len(df_disc)})")

# ----------------------------------------------------------------------
# 7) (Optional) Low-Confidence Sample Detection
# ----------------------------------------------------------------------
def detect_low_confidence_samples(model, dataset, threshold=0.6, label="Model"):
    """
    For each sample in the *subset* dataset, compute max probability.
    Return a DataFrame with those that have < threshold.

    Again we must carefully retrieve the row from dataset.dataset.df.
    """
    if not hasattr(dataset, 'indices'):
        print(f"[{label}] dataset is not a Subset with .indices attribute. Skipping low-confidence detection.")
        return pd.DataFrame()

    dl = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()

    subset_indices = dataset.indices  # the global indices
    full_df = dataset.dataset.df

    collected = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            local_idxs = batch["idx"].cpu().numpy()  # positions within subset
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            max_conf = np.max(probs, axis=1)

            for i, conf in enumerate(max_conf):
                if conf < threshold:
                    local_idx = local_idxs[i]
                    if local_idx >= 0 and local_idx < len(subset_indices):
                        global_idx = subset_indices[local_idx]
                        row_data = full_df.iloc[global_idx]
                        collected.append({
                            "Global_Index": global_idx,
                            "Local_Index_in_Subset": local_idx,
                            "Sentence": row_data["Sentence"],
                            "True_Label": row_data["Label"],
                            "Max_Confidence": conf
                        })

    df_noisy = pd.DataFrame(collected)
    return df_noisy

# ----------------------------------------------------------------------
# 8) Evaluate on Test Files
# ----------------------------------------------------------------------
def evaluate_on_test_files(model, test_files, model_name):
    for lang, filepath in test_files.items():
        test_df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip').dropna(subset=["Sentence"])
        test_df["Label"] = test_df["Label"].map(LABEL_MAPPING)

        # Make a dataset on the fly (not a random_split subset => no .indices needed here)
        tmp_dataset = TextDataset(test_df)
        dl = DataLoader(tmp_dataset, batch_size=16, shuffle=False)

        metrics, _, _ = evaluate_model(model, dl)
        wandb.log({
            f"test/{lang}_accuracy_{model_name}": metrics["accuracy"],
            f"test/{lang}_f1_{model_name}": metrics["f1"],
        })
        print(f"[{model_name}] Test {lang} metrics: {metrics}")

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    wandb.init(project="General-model-English", name="Sent-co-teaching", reinit=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ----------------- 1) Train with advanced co-teaching -----------------
    discarded_by_m1, discarded_by_m2 = train_co_teaching(
        model1, model2,
        train_dataloader, val_dataloader,
        optimizer1, optimizer2,train_dataset, 
        epochs=EPOCHS,
        rho_min=RHO_MIN,
        rho_max=RHO_MAX
    )

    # ----------------- 2) Write out final set of discarded points  -----------------
    # Each set is local indices within train_dataset. We map them to the original df.
    save_discarded_samples(
        discarded_indices=discarded_by_m1,
        dataset=train_dataset,  
        output_path=os.path.join(OUTPUT_DIR, "discarded_by_model1.csv"),
        label="Model1"
    )
    save_discarded_samples(
        discarded_indices=discarded_by_m2,
        dataset=train_dataset,
        output_path=os.path.join(OUTPUT_DIR, "discarded_by_model2.csv"),
        label="Model2"
    )

    # ----------------- 3) Optional: Detect final “low-confidence” samples  -----------------
    # Using the entire training subset for each model
    df_noisy_model1 = detect_low_confidence_samples(model1, train_dataset, threshold=CONFIDENCE_THRESHOLD, label="Model1")
    if not df_noisy_model1.empty:
        path_m1_noisy = os.path.join(OUTPUT_DIR, "low_confidence_model1.csv")
        df_noisy_model1.to_csv(path_m1_noisy, index=False)
        print(f"[Model1] Low-confidence samples saved to {path_m1_noisy} (n={len(df_noisy_model1)})")

    df_noisy_model2 = detect_low_confidence_samples(model2, train_dataset, threshold=CONFIDENCE_THRESHOLD, label="Model2")
    if not df_noisy_model2.empty:
        path_m2_noisy = os.path.join(OUTPUT_DIR, "low_confidence_model2.csv")
        df_noisy_model2.to_csv(path_m2_noisy, index=False)
        print(f"[Model2] Low-confidence samples saved to {path_m2_noisy} (n={len(df_noisy_model2)})")

    # ----------------- 4) Evaluate on Test Sets  -----------------

    print("\nLoading and Evaluating the Best Model (Model2) on test sets:")

    best_model2_path = os.path.join(OUTPUT_DIR, "best_model2.pth")
    model2.load_state_dict(torch.load(best_model2_path))
    model2.eval()

    evaluate_on_test_files(model2, TEST_FILES, model_name="best_model2")

    print(f"Best Model2 loaded from {best_model2_path}") 
    
    model1_save_path = os.path.join(OUTPUT_DIR, "co_teaching_model1.pth")
    model2_save_path = os.path.join(OUTPUT_DIR, "co_teaching_model2.pth")

    torch.save(model1.state_dict(), model1_save_path)
    torch.save(model2.state_dict(), model2_save_path)

    print(f"Model1 saved to {model1_save_path}")
    print(f"Model2 saved to {model2_save_path}")
    
    wandb.finish()
