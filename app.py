import os
import sys
import torch

# Temporary fix for streamlit-torch path error
sys.modules["torch.classes"] = None

import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence

# Optional: avoid USER_AGENT warning
os.environ["USER_AGENT"] = "UniversalSummarizer/1.0"

# Streamlit page config
st.set_page_config(page_title="Universal Summarizer", layout="centered")
st.title("📄 Universal Text Summarizer")

st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Paste or type your text (up to 3000 words).
2. Click **Summarize**.
3. View the summary below.
""")

# Load summarization pipeline using a simpler and robust model
@st.cache_resource
def load_model():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Simpler, works without special handling
        tokenizer="sshleifer/distilbart-cnn-12-6",
        device=-1,
        framework="pt"
    )
    return HuggingFacePipeline(pipeline=summarizer)

llm = load_model()
prompt = PromptTemplate.from_template("Summarize this:\n\n{text}\n\nSummary:")
summarization_chain = prompt | llm

MAX_CHARS = 4000  # Lowered for safety with tokenizer limits

# Input text box
text = st.text_area("Enter the text to summarize (up to 3000 words):", height=400)

# Summarize button
if st.button("🔍 Summarize"):
    if text.strip():
        if len(text) > MAX_CHARS:
            st.warning(f"Input too long ({len(text)} characters). Truncating to first {MAX_CHARS} characters.")
            text = text[:MAX_CHARS]
        with st.spinner("Summarizing..."):
            try:
                result = summarization_chain.invoke({"text": text})
                # If the result is a dict (as it often is), extract summary text
                summary = result if isinstance(result, str) else result.get("text", str(result))
                st.subheader("📝 Summary")
                st.success(summary)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
    else:
        st.warning("Please enter some text before summarizing.")
