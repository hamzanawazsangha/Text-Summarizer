import os
import sys
import torch

# Temporary fix for streamlit-torch path error
sys.modules["torch.classes"] = None

import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline  # Updated import
from langchain_core.runnables import RunnableSequence  # New chaining

# Optional: avoid USER_AGENT warning
os.environ["USER_AGENT"] = "UniversalSummarizer/1.0"

# Streamlit page config
st.set_page_config(page_title="Universal Summarizer", layout="centered")
st.title("📄 Universal Text Summarizer")

st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Paste or type your text.
2. Click **Summarize**.
3. View the summary below.
""")

MAX_CHARS = 2500  # Approximate max input characters to avoid embedding errors

# Load summarizer with truncation and max_length settings
@st.cache_resource
def load_model():
    pipe = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1,
        truncation=True,
        max_length=200,      # max tokens in output summary
        min_length=30,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# Prompt template
prompt = PromptTemplate.from_template("Summarize this:\n\n{text}\n\nSummary:")

# New LangChain Runnable format
summarization_chain = prompt | llm

# Input
text = st.text_area("Enter the text to summarize:", height=300)

if st.button("🔍 Summarize"):
    if text.strip():
        # Truncate input text if too long to avoid model errors
        if len(text) > MAX_CHARS:
            st.warning(f"Input too long, truncating to {MAX_CHARS} characters.")
            text = text[:MAX_CHARS]

        with st.spinner("Summarizing..."):
            result = summarization_chain.invoke({"text": text})
            st.subheader("📝 Summary")
            st.success(result)
    else:
        st.warning("Please enter some text before summarizing.")
