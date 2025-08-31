import spacy
from spacy.util import minibatch
import random
from spacy.training import Example
from spacy.tokens import DocBin
import os
import time

# Enable GPU if available
spacy.require_gpu()
print("GPU available:", spacy.prefer_gpu())

# Paths
CHECKPOINT_DIR = "drug_symptom_ner_checkpoint"
FINAL_MODEL_DIR = "drug_symptom_ner_model"
TRAIN_DATA_PATH = "combined_training_data.spacy"

# Load data
def load_training_data(file_path):
    nlp_blank = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    return list(doc_bin.get_docs(nlp_blank.vocab))

train_data = load_training_data(TRAIN_DATA_PATH)
train_data=train_data[:50000]


# Resume if checkpoint exists, else start fresh
if os.path.exists(CHECKPOINT_DIR):
    print("🔄 Resuming from checkpoint...")
    nlp = spacy.load(CHECKPOINT_DIR)
else:
    print("🚀 Starting fresh training...")
    nlp = spacy.load("en_core_web_sm")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    ner.add_label("DRUG")
    ner.add_label("SYMPTOM")

# Optimizer
optimizer = nlp.resume_training() if os.path.exists(CHECKPOINT_DIR) else nlp.create_optimizer()

# Training config
n_iter = 100
#BATCH_SIZE = 32
DROPOUT = 0.3

# Training loop
for epoch in range(n_iter):
    random.shuffle(train_data)
    losses = {}
    
    print(f"\n📌 Epoch {epoch+1}/{n_iter} started...")
    start_epoch = time.time()

    for batch_i, batch in enumerate(minibatch(train_data, size=1500), 1):
        batch_start = time.time()
        print(len(batch))

        for doc in batch:
            example = Example.from_dict(
                doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
            )
            nlp.update([example], drop=DROPOUT, sgd=optimizer, losses=losses)
        
        # Print every 1000 batches
        if batch_i % 20 == 0:
            batch_time = time.time() - batch_start
            print(f"   🔹 Epoch {epoch+1} | Batch {batch_i} | Loss: {losses.get('ner', 0):.4f} | Batch time: {batch_time:.2f}s")

    epoch_time = time.time() - start_epoch
    print(f"✅ Epoch {epoch+1}/{n_iter} completed | Loss: {losses.get('ner', 0):.4f} | Time: {epoch_time:.2f}s")

    # Save checkpoint every epoch
    nlp.to_disk(CHECKPOINT_DIR)

# Save final model
print("🎉 Training finished! Saving final model...")
nlp.to_disk(FINAL_MODEL_DIR)
