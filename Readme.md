# Adverse Drug Reaction Detection using DeBERTa-v3

A Named Entity Recognition (NER) system for detecting Adverse Drug Reaction (ADR) mentions in clinical and user-generated text, fine-tuned on DeBERTa-v3-base.

---

## Project Overview

Adverse Drug Reactions (ADRs) are a critical concern in pharmacovigilance. This project fine-tunes Microsoft's DeBERTa-v3-base model on annotated ADR datasets to automatically detect and classify ADR mentions at the token level using sequence labeling (NER).

This work was developed during a research internship at **CDAC Pune**.

---

## Task

- **Task Type:** Token Classification (Named Entity Recognition)
- **Model:** `microsoft/deberta-v3-base`
- **Objective:** Identify tokens in medical/social text that correspond to adverse drug reaction mentions

---

## Model Architecture

- **Base Model:** DeBERTa-v3-base (Disentangled Attention + Enhanced Mask Decoder)
- **Head:** Token Classification head (`AutoModelForTokenClassification`)
- **Max Sequence Length:** 128 tokens
- **Label Alignment:** Word-piece tokens aligned to word-level BIO tags; subword tokens assigned `-100` (ignored in loss)

---

## Dataset

### Base Dataset — BC5CDR (BioCreative V CDR)
- **Full Name:** BioCreative V Chemical-Disease Relation Dataset
- **Source:** publicly available via HuggingFace / BioCreative V challenge
- **License:** Open for research use
- **Size:** ~1,500 PubMed abstracts, ~15,000+ annotated entities (chemicals & diseases)
- **Labels:** BIO-tagged chemical and disease mentions
- **Format:** JSON files with tokenized sequences and BIO-tagged labels
- **Splits:** `train.json`, `valid.json`, `test.json`

### Data Augmentation — Synthetic Data
To address class imbalance and limited ADR-positive examples, the training set was expanded using **synthetic data generation**:
- Generated additional ADR mention sentences by paraphrasing and substituting drug/reaction entities from the original corpus
- Augmented samples were validated for label consistency before inclusion
- This improved model performance on minority ADR classes

### Data Cleaning Steps Performed
- Removed duplicate abstracts and near-duplicate token sequences
- Handled misaligned BIO tags caused by tokenization inconsistencies
- Normalized whitespace and special characters in raw text before tokenization

---

## Project Structure

```
Adversial-drug-reaction/
├── deberta_model.py       # Core model training and tokenization pipeline
├── train.json             # Training data (tokenized + BIO labels)
├── valid.json             # Validation data
├── test.json              # Test data
├── label.json             # Label list for NER tags
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Model

```bash
python deberta_model.py
```

The script will:
1. Load and tokenize the dataset from JSON files
2. Align BIO labels to DeBERTa subword tokens
3. Fine-tune `microsoft/deberta-v3-base` for token classification
4. Evaluate on validation set using seqeval metrics (F1, Precision, Recall)

---

## Key Implementation Details

### Tokenization & Label Alignment

DeBERTa uses subword tokenization. Since NER labels are at the word level, labels are aligned using `word_ids()`:
- First subword of each word → assigned the word's label
- Subsequent subwords and special tokens → assigned `-100` (ignored in cross-entropy loss)

```python
for word_idx in word_ids:
    if word_idx is None:
        label_ids.append(-100)
    elif word_idx != previous_word_idx:
        label_ids.append(label[word_idx])
    else:
        label_ids.append(-100)
```

### Why DeBERTa over alternatives?

| Model | Reason Not Chosen |
|---|---|
| BERT-base | Weaker on domain-specific medical text; older attention mechanism |
| RoBERTa | No disentangled positional attention; slightly lower NER benchmark scores |
| **DeBERTa-v3-base** ✅ | Disentangled attention captures both content and position separately — better for entity boundary detection |

---

## Dependencies

| Library | Purpose |
|---|---|
| `transformers` | DeBERTa model, tokenizer, Trainer API |
| `datasets` | Loading JSON data splits |
| `evaluate` | Metric computation |
| `seqeval` | NER-specific F1, precision, recall |
| `torch` | Deep learning backend |
| `numpy` | Numerical operations |

---

## Responsible AI

- **Dataset Bias:** ADR datasets are known to underrepresent reactions from elderly patients and non-English speakers due to pharmacovigilance reporting bias. Class-weight balancing was applied during training to partially mitigate majority-class dominance.
- **Model License:** DeBERTa-v3-base is released under MIT License via HuggingFace.
- **Repo License:** MIT — compatible with all dependencies used.

---

## Author

**Rishit Tiwari**
Research Intern, CDAC Pune