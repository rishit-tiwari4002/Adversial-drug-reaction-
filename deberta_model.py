from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from math import ceil
import numpy as np
import evaluate
import json
import os

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("json", data_files={
    "train": "dataset/train.json",   # JSONL format
    "validation": "dataset/valid.json",
    "test": "dataset/test.json"
})

# Load label mappings
with open("dataset/label-timeline.json", "r") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

print("Dataset columns:", dataset['train'].column_names)
print("Number of labels:", num_labels)

# -------------------------------
# Tokenizer & model
# -------------------------------
base_checkpoint = "microsoft/deberta-v3-base"   # start fresh from base
output_dir = "deberta_basic-timeline"                    # where checkpoints will be saved

tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)

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

model = AutoModelForTokenClassification.from_pretrained(
    base_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# Training setup
# -------------------------------
train_batch_size = 28
steps_per_epoch = ceil(len(tokenized_datasets["train"]) / train_batch_size)

args = TrainingArguments(
    output_dir="deberta_basic-timeline",
    eval_strategy="epoch",     # ✅ evaluate after each epoch
    save_strategy="steps",           # ✅ still saving every N steps
    save_steps=steps_per_epoch * 5,  # save every ~5 epochs worth of steps
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=28,
    num_train_epochs=20,
    weight_decay=0.03,
    logging_dir="./logs",
    logging_steps=100,
    # remove load_best_model_at_end since strategies don't match
)


# -------------------------------
# Metrics
# -------------------------------
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -------------------------------
# Resume if checkpoint exists
# -------------------------------
last_checkpoint = None
if os.path.isdir(output_dir):
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint:
    print(f"🔄 Resuming training from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("🚀 Starting fresh training from base model")
    trainer.train()

# -------------------------------
# Save final model
# -------------------------------
trainer.save_model("ner_model_timeline")
tokenizer.save_pretrained("ner_model_timeline")

print("✅ Training complete, model saved at ner_model_stage1")
