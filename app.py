import os
import sys
import torch
import types

# Safe workaround for torch.classes RuntimeError in Streamlit
if not hasattr(torch, "classes") or isinstance(torch.classes, types.ModuleType):
    class _TorchClassesMock:
        def __getattr__(self, name):
            return None
    sys.modules["torch.classes"] = _TorchClassesMock()

import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline  # Updated import
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

llm = load_model()
prompt = PromptTemplate.from_template("Summarize this:\n\n{text}\n\nSummary:")
summarization_chain = prompt | llm

MAX_CHARS = 22000  # Approx. 3000 words

text = st.text_area("Enter the text to summarize (up to 3000 words):", height=400)

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
