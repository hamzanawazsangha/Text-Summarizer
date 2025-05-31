import os
import streamlit as st
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline  # Updated import

# Optional: Set user agent to avoid warning
os.environ["USER_AGENT"] = "UniversalSummarizer/1.0"

# Set page config
st.set_page_config(page_title="Universal Summarizer", layout="centered")

# Title
st.title("📄 Universal Text Summarizer")

# Sidebar instructions
st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Paste or type the text you want to summarize.
2. Click the **Summarize** button.
3. The summary will appear below!
""")

# Cache the summarizer model
@st.cache_resource
def load_summarizer():
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    return HuggingFacePipeline(pipeline=summarizer_pipeline)

summarizer = load_summarizer()

# Prompt template for chaining
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize this:\n\n{text}\n\nSummary:"
)

# Create chain
summarization_chain = LLMChain(
    llm=summarizer,
    prompt=prompt_template
)

# User input
user_input = st.text_area("Enter the text to summarize:", height=300)

# Button to summarize
if st.button("🔍 Summarize"):
    if user_input.strip():
        with st.spinner("Generating summary..."):
            result = summarization_chain.run(user_input)
            st.subheader("📝 Summary")
            st.success(result)
    else:
        st.warning("Please enter some text before summarizing.")
