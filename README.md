
# Cross-Lingual Sentence Complexity Classification

This repository implements a cross-lingual sentence-difficulty classification framework in the **"From Babel to Brilliance: De-Noising Techniques for Cross-Lingual Sentence-Difficulty Classifiers"** . The framework integrates pre-trained language models, noise reduction techniques, and evaluation pipelines for cross-lingual classification tasks.


## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Evaluation](#evaluation)
7. [Citation](#citation)

---

## **Overview**
This project aims to improve the robustness and generalization of multilingual sentence-difficulty classifiers by:
- Leveraging pre-trained language models like XLM-RoBERTa and mBERT.
- Employing noise reduction techniques such as **Small-Loss Trick**, **Label Smoothing**, and **Noise Transition Matrices**.
- Enabling cross-lingual transfer learning to classify sentence difficulty across multiple languages.

### **Key Objectives**
1. Analyze the impact of noisy training data on sentence-level difficulty classification.
2. Identify optimal segment lengths and effective noise reduction techniques.
3. Provide a reusable, robust pipeline for cross-lingual sentence classification.

---

## **Features**
- **Pre-trained Models**: Supports models like `xlm-roberta` and `BERT`.
- **Noise Reduction Techniques**: Includes implementations of:
  - **Small-Loss Trick (ST)**
  - **Label Smoothing (LabS)**
  - **Co-Teaching (CT)**
  - **Noise Transition Matrices (NTM)**
- **Cross-Lingual Transfer**: Trains models on one language and evaluates performance on others.
- **Customizable Training**: Specify model, dataset, and hyperparameters via command-line arguments.
- **Metrics**: Calculates F1-score, precision, recall, ROC-AUC, log-loss, and more.

---

## **Setup**
### **Prerequisites**
- Python 3.8 or higher
- NVIDIA GPU (optional for GPU-based training)
- Libraries:
  - `transformers`
  - `torch`
  - `datasets`
  - `wandb`
  - `scikit-learn`
  - `numpy`
  - `scipy`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/cross-lingual-sentence-classification.git
   cd cross-lingual-sentence-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Log into [Weights & Biases](https://wandb.ai/) for experiment tracking:
   ```bash
   wandb login
   ```

---

## **Usage**

### **Training**
Train a model with the following command:
```bash
python main.py -i <path_to_training_data> -m <model_name> -p <project_name> --batch_size 16 --epochs 4
```

Example:
```bash
python main.py -i data/training.csv -m xlm-roberta-base -p ComplexityProject --valsplit 0.2
```

### **Evaluation**
Evaluate a pre-trained model on a test dataset:
```bash
python main.py -t <path_to_test_data> -l <model_dir>
```

Example:
```bash
python main.py -t data/test.csv -l models/bert-multilingual-trained
```

### **Command-Line Arguments**
| Argument           | Description                                   | Default               |
|---------------------|-----------------------------------------------|-----------------------|
| `-m, --mname`      | Hugging Face model name                      | `bert-base-multilingual-cased`    |
| `-l, --local`      | Path to the local trained model               | `None`                |
| `-p, --projectname`| Weights & Biases project name                | `None`                |
| `-i, --inputfile`  | Path to the training dataset                  | `None`                |
| `-t, --testfile`   | Path to the test dataset                      | `None`                |
| `--lr`             | Learning rate for the optimizer               | `1e-5`                |
| `-e, --epochs`     | Number of training epochs                     | `4`                   |
| `--batch_size`     | Batch size for training                       | `16`                  |
| `--valsplit`       | Validation split ratio                        | `0.2`                 |

---

## **Methodology**
The methodology follows the framework described in the COLING 2025 paper:
1. **Text Segmentation**: Texts are split into segments of varying lengths (e.g., 50, 100 tokens).
2. **Noise Reduction**:
   - **Small-Loss Trick (ST)**: Filters out high-loss samples.
   - **Label Smoothing (LabS)**: Reduces overfitting by redistributing probabilities.
   - **Noise Transition Matrices (NTM)**: Models and corrects noisy labels.
3. **Model Training**: Fine-tunes pre-trained models using a Hugging Face Trainer pipeline.
4. **Cross-Lingual Evaluation**: Evaluates trained models on multiple target languages.

---

## **Evaluation**
Use the provided datasets to evaluate the model. Key evaluation metrics include:
- **Accuracy**
- **F1-Score**
- **Precision/Recall**
- **ROC-AUC**
- **Log-Loss**
- **Prediction Entropy**

Results are saved as a CSV file for further analysis.

---

## **Citation**
If you use this repository or methodology in your research, please cite the paper:

```bibtex
@inproceedings{babel_to_brilliance,
  title={From Babel to Brilliance: De-Noising Techniques for Cross-Lingual Sentence-Difficulty Classifiers},
  author={Nouran Khallaf, Serge Sharoff},
  year={2025},
}
```
