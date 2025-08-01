import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Fixes TF model conflict

import re
import nltk
import json
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from IPython.display import Audio, display
import language_tool_python
nltk.download("punkt")
from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]

if filename.endswith(".pdf"):
    with fitz.open(filename) as pdf:
        raw_text = ""
        for page in pdf:
            raw_text += page.get_text()
else:
    raw_text = uploaded[filename].decode("utf-8")

print("📄 Document loaded successfully!")
def split_into_clauses(text):
    clause_pattern = r"(Clause\s*\d+[:.\-]?\s*.?)(?=Clause\s\d+[:.\-]?|\Z)"
    matches = re.findall(clause_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return [clause.strip() for clause in matches if clause.strip()]
    alternate_pattern = r"((?:Section|Clause)?\s*\d+[:.\-]?\s*.?)(?=(?:Section|Clause)?\s\d+[:.\-]?|\Z)"
    matches = re.findall(alternate_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches and len(matches) > 1:
        return [clause.strip() for clause in matches if clause.strip()]
    return [p.strip() for p in text.split("\n\n") if p.strip()]

# Load Simplifier Model
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def simplify_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=60,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_document(text):
    labels = ["NDA", "Contract", "Agreement", "Lease", "Employment Agreement", "License", "Terms and Conditions"]
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]

def assess_risk(text):
    result = classifier(text, candidate_labels=["risky", "not risky", "safe", "high risk", "low risk"])
    for label, score in zip(result["labels"], result["scores"]):
        if label in ["risky", "high risk"]:
            return f"⚠️ High Risk ({label})", score
        elif label in ["safe", "not risky", "low risk"]:
            return f"✅ Low Risk ({label})", score
    return "Unknown", 0
print("📘 Document Classification:")
doc_type, doc_score = classify_document(raw_text)
print(f"Predicted: {doc_type} (Confidence: {doc_score:.2f})")

clauses = split_into_clauses(raw_text)
print(f"\n📄 Found {len(clauses)} clauses.")

for i, clause in enumerate(clauses, 1):
    print(f"\n🔸 Clause {i}:\n{clause}")
    if len(clause.split()) > 30:
        simplified = simplify_text(clause)
        print(f"✅ Simplified:\n{simplified}")
    else:
        print("⚠️ Too short to simplify.")

risk_label, risk_score = assess_risk(raw_text)
print(f"\n📊 Risk Assessment: {risk_label} (Confidence: {risk_score:.2f})")

