import os
import streamlit as st
import torch
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence

# Set custom user agent to avoid warning
os.environ["USER_AGENT"] = "UniversalSummarizer/1.0"

# Streamlit page configuration
st.set_page_config(page_title="Universal Summarizer", layout="centered")
st.title("📄 Universal Text Summarizer")

st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Paste or type your text (up to 3000 words).
2. Click **Summarize**.
3. View the summary below.
""")

@st.cache_resource
def load_model():
    summarizer = pipeline(
        "summarization",
        model="allenai/led-large-16384",
        tokenizer="allenai/led-large-16384",
        device=-1,  # CPU
        framework="pt",
        torch_dtype=torch.float32,
        truncation=True,
        max_length=512,
        min_length=30,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=summarizer)

# Load model
llm = load_model()

# Setup summarization prompt
prompt = PromptTemplate.from_template("Summarize this:\n\n{text}\n\nSummary:")
summarization_chain = prompt | llm

# 3000 words ≈ 22,000 characters
MAX_CHARS = 22000

# Input text box
text = st.text_area("Enter the text to summarize (up to 3000 words):", height=400)

# Button to trigger summarization
if st.button("🔍 Summarize"):
    if text.strip():
        if len(text) > MAX_CHARS:
            st.warning(f"Input too long ({len(text)} characters). Truncating to first {MAX_CHARS} characters (~3000 words).")
            text = text[:MAX_CHARS]
        with st.spinner("Summarizing..."):
            result = summarization_chain.invoke({"text": text})
            st.subheader("📝 Summary")
            st.success(result)
    else:
        st.warning("Please enter some text before summarizing.")
