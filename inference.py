from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Path to your saved model
model_dir = "ner_model_timeline"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Create pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Read the text file
with open("discharge_summary.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Extracted text:", text[:400], "...")  # preview first 200 chars

# Run NER
ner_results = ner_pipeline(text)

# Show results
for entity in ner_results:
    print(f"{entity['word']} → {entity['entity_group']} ({entity['score']:.2f})")

with open("ner_results.txt", "w", encoding="utf-8") as f:
    for entity in ner_results:
        f.write(f"{entity['word']} → {entity['entity_group']} ({entity['score']:.2f})\n")
