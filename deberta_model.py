from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModel
import numpy as np
import evaluate
import json

# Load dataset splits
dataset = load_dataset("json", data_files={
    "train": "train.json",
    "validation": "valid.json",
    "test": "test.json"
})

# Load label list
with open("label.json", "r") as f:
    label_list = json.load(f)

print(dataset['train'].column_names)

model_checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["tags"]):  # labels are already ints
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # directly use int tag
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

num_labels = len(label_list)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels
)