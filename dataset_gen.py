import pandas as pd
import spacy
from spacy.tokens import DocBin

if spacy.require_gpu():
    print("✅ GPU enabled!")
else:
    print("⚠️ GPU not available, running on CPU")

# --------- Load your drug and symptom lists ----------
# Drugs CSV with multiple columns
drug_df = pd.read_csv("drugs.csv", dtype=str)   # <--- force read as string
drug_list = drug_df.values.flatten().tolist()
drug_list = [str(d).strip() for d in drug_list if pd.notna(d)]

# Symptoms CSV with one column
symptom_df = pd.read_csv("symtoms2.csv", dtype=str)  # <--- force read as string
symptom_list = symptom_df.iloc[:,0].dropna().tolist()
symptom_list = [str(s).strip() for s in symptom_list]

nlp = spacy.blank("en")
db = DocBin()

# --------- Templates ----------
drug_templates = [
    "The patient was prescribed {}.",
    "{} was given to the patient for treatment.",
    "The doctor recommended {} during the visit."
]

symptom_templates = [
    "The patient reported {}.",
    "He is suffering from {}.",
    "{} was noted in the medical report."
]

# --------- Function to generate training data ----------
def add_examples(entity_list, label, templates):
    for item in entity_list:
        item = str(item)  # <--- make sure it's a string
        for template in templates:
            text = template.format(item)
            doc = nlp.make_doc(text)
            start = text.find(item)
            end = start + len(item)
            span = doc.char_span(start, end, label=label)
            if span:
                doc.ents = [span]
                db.add(doc)

# --------- Generate ----------
add_examples(drug_list, "DRUG", drug_templates)
add_examples(symptom_list, "SYMPTOM", symptom_templates)

# --------- Save final training data ----------
db.to_disk("combined_training_data.spacy")
print("✅ Training data saved as combined_training_data.spacy")
