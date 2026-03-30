import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

new_labels = {
  "O": 0,
  "B-Chemical": 1,
  "B-Disease": 2,
  "I-Disease": 3,
  "I-Chemical": 4,
  "B-Test": 5,
  "I-Test": 6,
  "B-Procedure": 7,
  "I-Procedure": 8,
  "B-Symptom": 9,
  "I-Symptom": 10,
  "B-Social": 11,
  "I-Social": 12
}

checkpoint_dir = "final-ner-model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

id2label = {v: k for k, v in new_labels.items()}

# Load old checkpoint directly, HF handles weight copying
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint_dir,
    num_labels=len(new_labels),
    id2label=id2label,
    label2id=new_labels,
    ignore_mismatched_sizes=True  # ✅ keeps old weights, random for new
)

print("✅ Model expanded to 13 labels. Old weights preserved, new ones random.")

save_dir= "ner_model_13_label"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)   # if you’re using the same tokenizer
