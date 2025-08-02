import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
import re
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="ClauseWise AI", layout="wide")
st.title("📜 ClauseWise AI – Smart Legal Analyzer")

uploaded_file = st.file_uploader("📁 Upload your Legal Document (PDF or TXT)", type=["pdf", "txt"])

@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return summarizer_tokenizer, summarizer_model, classifier

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def split_into_clauses(text):
    pattern = r"(Clause\s+\d+:\s+.*?)(?=\nClause\s+\d+:|\Z)"
    clauses = re.findall(pattern, text, flags=re.DOTALL)
    return [c.strip() for c in clauses if len(c.strip().split()) > 3]

def classify_document(classifier, text):
    labels = ["NDA", "Contract", "Agreement", "Lease", "Employment Agreement", "License", "Terms and Conditions"]
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0], result["scores"][0]

def simplify_clause(tokenizer, model, text):
    cleaned = re.sub(r"^Clause\s*\d+[:\-]?\s*", "", text).strip()
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def assess_risk(classifier, text):
    labels = ["risky", "safe", "not risky", "high risk", "low risk"]
    result = classifier(text, candidate_labels=labels)
    for label, score in zip(result["labels"], result["scores"]):
        if label in ["risky", "high risk"]:
            return f"⚠️ High Risk ({label})", score
        elif label in ["safe", "not risky", "low risk"]:
            return f"✅ Low Risk ({label})", score
    return "⚠️ Unknown Risk", 0

if uploaded_file:
    st.success("📄 File Uploaded Successfully")

    raw_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_txt(uploaded_file)

    with st.expander("🔍 Raw Extracted Text"):
        st.write(raw_text)

    tokenizer, model, classifier = load_models()

    with st.spinner("🔎 Classifying document..."):
        doc_type, doc_conf = classify_document(classifier, raw_text)
    st.info(f"📘 Document Type: {doc_type} (Confidence: {doc_conf:.2f})")

    clauses = split_into_clauses(raw_text)
    st.success(f"🧩 Detected {len(clauses)} clause(s)")

    for i, clause in enumerate(clauses, 1):
        st.subheader(f"📌 Clause {i}")
        st.code(clause)

        if len(clause.split()) > 20:
            with st.spinner("✍ Simplifying..."):
                simplified = simplify_clause(tokenizer, model, clause)
            st.success("✅ Simplified:")
            st.write(simplified)

            risk_label, risk_score = assess_risk(classifier, clause)
            if "High Risk" in risk_label:
                st.error(f"📊 Risk: {risk_label} (Confidence: {risk_score:.2f})")
            else:
                st.success(f"📊 Risk: {risk_label} (Confidence: {risk_score:.2f})")
        else:
            st.warning("⚠️ Too short to simplify or assess.")
else:
    st.info("👈 Upload a file to begin.")
