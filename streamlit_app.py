import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow usage
os.environ["USE_TF"] = "0"              # Prevent fallback to TF

import streamlit as st
import re
import fitz  # PyMuPDF for reading PDFs

from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM




st.set_page_config(page_title="ClauseWise AI", layout="wide")
st.title("📜 ClauseWise AI – Smart Legal Analyzer")

uploaded_file = st.file_uploader("📁 Upload your Legal Document (PDF or TXT)", type=["pdf", "txt"])

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

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

def simplify_text(tokenizer, model, text):
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

def classify_document(classifier, text):
    labels = ["NDA", "Contract", "Agreement", "Lease", "Employment Agreement", "License", "Terms and Conditions"]
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]

def assess_risk(classifier, text):
    labels = ["risky", "not risky", "safe", "high risk", "low risk"]
    result = classifier(text, candidate_labels=labels)
    for label, score in zip(result["labels"], result["scores"]):
        if label in ["risky", "high risk"]:
            return f"⚠️ High Risk ({label})", score
        elif label in ["safe", "not risky", "low risk"]:
            return f"✅ Low Risk ({label})", score
    return "Unknown", 0

if uploaded_file:
    st.success("📄 File Uploaded Successfully")

    raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_txt(uploaded_file)

    with st.expander("🔍 Raw Extracted Text"):
        st.write(raw_text)

    tokenizer, model = load_summarizer()
    classifier = load_classifier()

    with st.spinner("🔎 Classifying document..."):
        doc_type, doc_conf = classify_document(classifier, raw_text)
    st.info(f"📘 Document Classification: {doc_type} (Confidence: {doc_conf:.2f})")

    clauses = split_into_clauses(raw_text)
    st.success(f"🧩 Found {len(clauses)} clauses.")

    for i, clause in enumerate(clauses, 1):
        st.subheader(f"📌 Clause {i}")
        st.code(clause)
        if len(clause.split()) > 30:
            with st.spinner("✍️ Simplifying..."):
                simplified = simplify_text(tokenizer, model, clause)
            st.success("✅ Simplified:")
            st.write(simplified)
        else:
            st.warning("⚠️ Too short to simplify.")

    risk_label, risk_score = assess_risk(classifier, raw_text)
    st.markdown("---")
    if "High Risk" in risk_label:
        st.error(f"📊 Risk Assessment: {risk_label} (Confidence: {risk_score:.2f})")
    elif "Low Risk" in risk_label:
        st.success(f"📊 Risk Assessment: {risk_label} (Confidence: {risk_score:.2f})")
    else:
        st.warning("📊 Risk Assessment: Unable to determine.")

else:
    st.info("👈 Upload a file to begin.")


