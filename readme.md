# Benchmarking Text Encoding Strategies for Surgical Case Duration Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Journal: IJMI](https://img.shields.io/badge/Journal-Int.%20J.%20Medical%20Informatics-green)](https://www.sciencedirect.com/journal/international-journal-of-medical-informatics)

This repository contains the source code for the paper:

> **Benchmarking Text Encoding Strategies in Multimodal Clinical Data for Surgical Case Duration Prediction**
> Mohammad Noorchenarboo, Michelle Kwong, Ahmad Elnahas, Jeff Hawel, Christopher M. Schlachta, Katarina Grolinger
> *International Journal of Medical Informatics* (submitted)

---

## Overview

Accurate prediction of surgical case duration is critical for OR scheduling, resource allocation, and patient throughput. This work systematically benchmarks a range of text encoding strategies — including classical methods (Label Encoding, TF-IDF, Count Vectorizer) and contextual embeddings (ClinicalBERT, Sentence-BERT) — applied to multimodal clinical data combining structured patient and procedural features with free-text clinical descriptions.

Six regression models are evaluated within a 5-fold cross-validation framework: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, and a Multi-Layer Perceptron (MLP).

---

## Repository Structure

```
├── 01. Pre-processing.py     # Data cleaning, feature engineering, fold generation
├── 02. BERT_Cache.py         # ClinicalBERT and Sentence-BERT embedding computation
├── 03. Fold_Encoding.py      # Fold-wise feature encoding (label, TF-IDF, count)
└── 04. Modeling.py           # HPO, model training, evaluation, and result logging
```

---

## Pipeline

The four scripts must be run sequentially.

### 1. Pre-processing (`01. Pre-processing.py`)
- Loads the raw input CSV and performs datetime feature derivation, duration filtering, categorical cleaning, and one-hot encoding
- Interactively prompts the user to select which text columns to activate (`scheduled_procedure`, `procedure`, `operative_dx`, `most_responsible_dx`)
- Saves the cleaned dataset and 5-fold cross-validation indices to a SQLite database

### 2. BERT Embedding Cache (`02. BERT_Cache.py`)
- Computes and caches contextual embeddings for each active text column using **ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) and **Sentence-BERT** (`all-MiniLM-L6-v2`)
- Saves one `.npy` file per model × column combination
- Automatically detects active text columns from the metadata written by Step 1

### 3. Fold Encoding (`03. Fold_Encoding.py`)
- Applies fold-wise XGBoost-based imputation for age, BMI, and anesthetic type (fit on training data only)
- Encodes text features using Label, TF-IDF, and Count Vectorizer at configurable feature counts (default: 10, 50, 100, 200)
- Supports incremental runs — already-computed encoding keys are skipped automatically

### 4. Modeling (`04. Modeling.py`)
- Trains and evaluates six regression models across all encoding strategies
- Applies Optuna-based hyperparameter optimization (5 trials per combination)
- Applies fold-wise PCA dimensionality reduction for BERT encodings (fit on training data only)
- Saves metrics, predictions, feature importances, hyperparameters, and timing to a result database
- Prints final summary tables comparing all encoding strategies across folds

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost sentence-transformers transformers torch tensorflow optuna
```

GPU acceleration is supported for BERT embedding computation and MLP training. Mixed precision (float16) is enabled automatically when a compatible GPU is detected.

---

## Reproducibility

All random seeds are fixed (`RANDOM_STATE = 42`). The 5-fold split is generated deterministically using `sklearn.model_selection.KFold(shuffle=True, random_state=42)` and stored in the database to ensure identical splits across all encoding and modeling stages. Full experimental results and analysis are reported in the paper.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{noorchenarboo2025surgical,
  title   = {Benchmarking Text Encoding Strategies in Multimodal Clinical Data for Surgical Case Duration Prediction},
  author  = {Noorchenarboo, Mohammad and Kwong, Michelle and Elnahas, Ahmad and Hawel, Jeff and Schlachta, Christopher M. and Grolinger, Katarina},
  journal = {International Journal of Medical Informatics},
  year    = {2025}
}
```

---

## Authors

| Author | Affiliation |
|---|---|
| Mohammad Noorchenarboo | Dept. of Electrical and Computer Engineering, Western University, London, Canada |
| Michelle Kwong | Dept. of Anesthesiology and Pain Medicine, University of Alberta; Dept. of Medicine, Western University |
| Ahmad Elnahas | Dept. of Surgery, Western University; London Health Sciences Centre |
| Jeff Hawel | Dept. of Surgery, Western University |
| Christopher M. Schlachta | Dept. of Surgery, Western University |
| Katarina Grolinger *(Corresponding)* | Dept. of Electrical and Computer Engineering, Western University — kgroling@uwo.ca |

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
