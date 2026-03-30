from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
import json
from datasets import load_dataset
import numpy as np
import evaluate

# Load dataset
dataset = load_dataset("json", data_files={"test": "dataset/test.json"})

# Load label mapping
with open("dataset/label.json", "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# Load tokenizer and model from checkpoint folder
model_checkpoint = "ner_model_stage1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)


# Tokenize
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=100
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Metrics
metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Evaluate
trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
metrics = trainer.evaluate(tokenized_datasets["test"])
print(metrics)

trainer.save_model("ner_model_stage1")
tokenizer.save_pretrained("ner_model_stage1")
print("model is saved")