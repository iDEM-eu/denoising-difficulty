import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, brier_score_loss, roc_auc_score
from scipy.stats import entropy
import time
import numpy as np
import wandb
from utils.metrics_utils import compute_generalization_metrics, compute_metrics

# Load your datasets
df = pd.read_csv('./combined_data/200-token/sbert_isnoisy_combined_data_200_fr.ol', sep='\t', on_bad_lines='skip')
wandb.init(project="General-model-French", name="ntm_sBERT")
# Map string labels to integers
label_mapping = {"wiki": 1, "vikidia": 0, 1: 1, 0: 0}
df['Label'] = df['Label'].map(label_mapping)

# Noise Transition Matrix
df['Noisy_Label'] = df.apply(lambda x: 'noisy' if x['is_noisy'] else x['Label'], axis=1)
unique_labels = df['Label'].unique()
conf_matrix = pd.crosstab(df['Label'], df['Noisy_Label'], rownames=['True'], colnames=['Noisy'], normalize='index')
conf_matrix = conf_matrix.reindex(index=unique_labels, columns=unique_labels, fill_value=0)
T = torch.tensor(conf_matrix.values, dtype=torch.float32)

epsilon = 1e-6
T = T * (1 - epsilon) + epsilon / T.size(1)

# Calculate inverse of the noise transition matrix
T_inv = torch.inverse(T)

print("Noise Transition Matrix T_inv:")
print(T_inv)

# Prepare the Dataset and Dataloaders
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx][0], self.texts[idx][1], self.labels[idx]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize_data(df):
    inputs = tokenizer(
        df['Sentence'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    return inputs

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_inputs = tokenize_data(train_df)
val_inputs = tokenize_data(val_df)

train_dataset = TextDataset(
    list(zip(train_inputs['input_ids'], train_inputs['attention_mask'])),
    torch.tensor(train_df['Label'].values, dtype=torch.long)
)
val_dataset = TextDataset(
    list(zip(val_inputs['input_ids'], val_inputs['attention_mask'])),
    torch.tensor(val_df['Label'].values, dtype=torch.long)
)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=16)

# Define the Model and Optimizer
model_noise_matrix = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(df['Label'].unique()))
optimizer_noise_matrix = AdamW(model_noise_matrix.parameters(), lr=2e-5)

# Adjust predictions using the inverse noise transition matrix
def adjust_predictions_with_noise_matrix(predictions, T_inv):
    adjusted_predictions = torch.matmul(predictions, T_inv.to(predictions.device))
    return torch.clamp(adjusted_predictions, min=0, max=1)  # Ensure probabilities are within valid range

# Save noisy sentences
def save_noisy_sentences(train_df, logits, output_dir='./NTM-EN-sBERT'):
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    noise_threshold = 0.8  # Adjust the threshold to a more lenient value

    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sample: {probabilities[:5]}")

    noise_mask = probabilities.max(axis=1) < noise_threshold

    print(f"Noise mask (first 10): {noise_mask[:10]}")
    print(f"Number of noisy sentences detected: {noise_mask.sum()}")

    if noise_mask.sum() > 0:
        noisy_sentences = train_df.iloc[noise_mask]
        noisy_sentences_file = f"{output_dir}/noisy_sentences.csv"
        noisy_sentences.to_csv(noisy_sentences_file, index=False)
        print(f"Noisy sentences saved to {noisy_sentences_file}")
    else:
        print("No noisy sentences detected. The CSV file will not be created.")

# Training loop with noise matrix adjustment
def train_with_noise_matrix(model, train_dataloader, optimizer, val_dataloader, epochs=3, output_dir="."):
    model.to(device)
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            adjusted_logits = adjust_predictions_with_noise_matrix(torch.softmax(logits, dim=1), T_inv)
            loss = torch.nn.functional.cross_entropy(adjusted_logits, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item()}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Detect and save noisy sentences
    print("Detecting noisy sentences...")
    all_logits = []

    # Process batches one by one to reduce memory usage
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())  # Move logits to CPU immediately to save GPU memory

    logits = torch.cat(all_logits, dim=0)  # Concatenate logits on the CPU
    save_noisy_sentences(train_df, logits, output_dir)

    return training_time

# Evaluate model and save results for each language
def evaluate_model_and_save(model, val_dataloader, test_files, T_inv):
    results = []

    def evaluate(dataloader):
        model.eval()
        all_labels = []
        all_preds = []
        all_losses = []
        all_logits = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                adjusted_logits = adjust_predictions_with_noise_matrix(torch.softmax(logits, dim=1), T_inv)
                preds = torch.argmax(adjusted_logits, dim=-1)
                loss = torch.nn.functional.cross_entropy(adjusted_logits, labels, reduction='none')
                probabilities = torch.softmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_losses.extend(loss.cpu().numpy())
                all_logits.extend(adjusted_logits.cpu().numpy())
                 # Store results properly
                all_probs.extend(probabilities.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        probabilities = np.array(all_logits)
        log_loss_value = log_loss(all_labels, probabilities)
        brier_score = brier_score_loss(all_labels, probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0])
        prediction_entropy = np.mean(entropy(probabilities, axis=1))
       # roc_auc = roc_auc_score(labels.cpu().numpy(), probabilities[:, 1]) if probabilities.shape[1] > 1 else None
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1]) if all_probs.shape[1] > 1 else None

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Log Loss": log_loss_value,
            "Brier Score": brier_score,
            "roc_auc" :roc_auc,
            "Prediction Entropy": prediction_entropy,
        }

    # Evaluate on validation data
    print("\nEvaluating on validation data:")
    val_results = evaluate(val_dataloader)
    results.append(["Validation"] + list(val_results.values()))

    # Evaluate on test files
    for lang, filepath in test_files.items():
        print(f"\nEvaluating on {lang} test data:")
        df_test = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
        test_inputs = tokenize_data(df_test)
        labels = torch.tensor(df_test['Label'].values, dtype=torch.long)

        test_dataset = DataLoader(
            TextDataset(
                list(zip(test_inputs['input_ids'], test_inputs['attention_mask'])),
                labels
            ),
            batch_size=16, shuffle=False
        )

        test_results = evaluate(test_dataset)
        results.append([lang] + list(test_results.values()))

    # Save results to a CSV file
    if len(results[0]) == 9:
        results_df = pd.DataFrame(results, columns=[
            "Dataset", "Accuracy", "Precision", "Recall", "F1", "Log Loss", "Brier Score", "ROC AUC", "Prediction Entropy"
           ])
    else:
        results_df = pd.DataFrame(results, columns=[
            "Dataset", "Accuracy", "Precision", "Recall", "F1", "Log Loss", "Brier Score", "Prediction Entropy"
         ])
    results_df.to_csv("./NTM-en-BERT-en/NTM_evaluation_results.csv", index=False)
    print("Evaluation results saved to NTM_evaluation_results.csv")

# Save the best model
def save_best_model(model):
    model_path = './NTM-fr-BERT-fr/NTM_best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_noise_matrix.to(device)

# Train the model with noise matrix adjustment
training_time = train_with_noise_matrix(model_noise_matrix, train_dataloader, optimizer_noise_matrix, val_dataloader, output_dir=".")

# List of test files to evaluate o
test_files = {
         "ca": "./combined_data/200-token/combined_data_200_ca.ol",
        "es": "./combined_data/200-token/combined_data_200_es.ol",
        "en": "./combined_data/200-token/combined_data_200_en.ol",
        "it": "./combined_data/200-token/combined_data_200_it.ol",
        "ru": "./combined_data/200-token/combined_data_200_ru.ol",

      } 


evaluate_model_and_save(model_noise_matrix, val_dataloader, test_files, T_inv)

# Save the trained model
save_best_model(model_noise_matrix)
