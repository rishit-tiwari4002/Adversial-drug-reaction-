import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # needed for sentence splitting

# ------------------------------
# STEP 1: Load and flatten drug list
# ------------------------------
def load_drugs(file_path):
    df = pd.read_csv(file_path)
    drug_set = set()
    for col in df.columns:
        drug_set.update(df[col].dropna().str.lower().str.strip())
    return drug_set

# ------------------------------
# STEP 2: Load and flatten symptom list
# ------------------------------
def load_symptoms(file_path):
    df = pd.read_csv(file_path)
    symptom_set = set()
    for col in df.columns:
        symptom_set.update(df[col].dropna().str.lower().str.strip())
    return symptom_set

# ------------------------------
# STEP 3: Tokenizer
# ------------------------------
def tokenize(sentence):
    return re.findall(r"\w+|[.,!?;]", sentence)

# ------------------------------
# STEP 4: BIO tagging function
# ------------------------------
def bio_tag_sentence(sentence, drug_set, symptom_set):
    tokens = tokenize(sentence)
    tags = ["O"] * len(tokens)

    for i, token in enumerate(tokens):
        word = token.lower()
        if word in drug_set:
            tags[i] = "B-DRUG"
        elif word in symptom_set:
            tags[i] = "B-SYMPTOM"

    return list(zip(tokens, tags))

# ------------------------------
# STEP 5: Process MIMIC NOTEEVENTS
# ------------------------------
def process_mimic_notes(noteevents_file, drug_set, symptom_set, output_file, limit=10000):
    df = pd.read_csv(noteevents_file, usecols=["TEXT"])
    print(f"Loaded {len(df)} notes from NOTEEVENTS.csv")

    with open(output_file, "w", encoding="utf-8") as f_out:
        count = 0
        for text in df["TEXT"].dropna():
            sentences = sent_tokenize(text)
            for sentence in sentences:
                token_tags = bio_tag_sentence(sentence, drug_set, symptom_set)
                for token, tag in token_tags:
                    f_out.write(f"{token} {tag}\n")
                f_out.write("\n")  # sentence boundary
            count += 1
            if limit and count >= limit:
                break  # safety limit so you don’t dump all 2M notes at once

    print(f"✅ Created NER dataset: {output_file} with {count} notes processed")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    drug_set = load_drugs("drugs.csv")
    symptom_set = load_symptoms("symptoms.csv")

    print(f"Loaded {len(drug_set)} unique drug names")
    print(f"Loaded {len(symptom_set)} unique symptoms")

    process_mimic_notes(
        noteevents_file="NOTEEVENTS.csv",
        drug_set=drug_set,
        symptom_set=symptom_set,
        output_file="ner_dataset.conll",
        limit=10000  # adjust/remove as needed
    )
