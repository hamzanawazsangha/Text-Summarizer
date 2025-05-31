import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    PyPDFLoader,
    Docx2txtLoader
)
import speech_recognition as sr
from langdetect import detect
from googletrans import Translator
import io
import os
from tempfile import NamedTemporaryFile

# --- Page Setup ---
st.set_page_config(
    page_title="Universal Text Summarizer",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- Header with Logo ---
col1, col2 = st.columns([1, 9])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/OpenAI_Logo.svg/512px-OpenAI_Logo.svg.png", width=60)
with col2:
    st.title("Universal Multilingual Text Summarizer")
    st.caption("Built with 💡 LangChain, 🌐 Multilingual Support, and 🧠 Streamlit")

# --- Environment Setup ---
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# --- Model Loader ---
@st.cache_resource
def load_summarizer():
    return HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={"temperature": 0, "max_length": 300}
    )

llm = load_summarizer()
translator = Translator()

# --- Text Processing ---
def process_text(text, word_limit, mode="map_reduce"):
    if not text.strip():
        return "❗ Please enter some text."

    try:
        # Detect and translate non-English text
        language = detect(text)
        if language != 'en':
            text = translator.translate(text, src=language, dest='en').text

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents([text])
        
        # Configure summarization chain
        chain = load_summarize_chain(
            llm,
            chain_type=mode.lower(),
            verbose=False
        )
        
        # Generate summary
        summary = chain.run(texts)
        
        # Translate back if needed
        if language != 'en':
            summary = translator.translate(summary, src='en', dest=language).text
        
        return summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Input Handlers ---
def handle_youtube(url):
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        return loader.load()[0].page_content
    except Exception as e:
        return f"Error: {str(e)}"

def handle_web_url(url):
    try:
        loader = WebBaseLoader(url)
        return loader.load()[0].page_content
    except Exception as e:
        return f"Error: {str(e)}"

def handle_pdf(file):
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        os.unlink(tmp.name)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error: {str(e)}"

def handle_docx(file):
    try:
        with NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.getvalue())
            loader = Docx2txtLoader(tmp.name)
            docs = loader.load()
        os.unlink(tmp.name)
        return docs[0].page_content
    except Exception as e:
        return f"Error: {str(e)}"

def handle_audio(file):
    recognizer = sr.Recognizer()
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.getvalue())
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
        os.unlink(tmp.name)
        return text
    except Exception as e:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return f"Error: {str(e)}"

# --- UI Components ---
st.sidebar.header("Upload Options")
input_option = st.sidebar.selectbox(
    "Choose input type:", 
    ["Text", "YouTube URL", "Article URL", "PDF", "DOCX", "Audio"]
)

input_text = ""

if input_option == "Text":
    input_text = st.text_area("📝 Enter text to summarize (any language):", height=300)
elif input_option == "YouTube URL":
    video_url = st.text_input("🎥 Enter YouTube Video URL:")
    if video_url:
        input_text = handle_youtube(video_url)
        if not input_text.startswith("Error"):
            st.success("YouTube transcript loaded successfully!")
elif input_option == "Article URL":
    article_url = st.text_input("📖 Enter Web Article URL:")
    if article_url:
        input_text = handle_web_url(article_url)
        if not input_text.startswith("Error"):
            st.success("Article content loaded successfully!")
elif input_option == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        input_text = handle_pdf(uploaded_pdf)
        if not input_text.startswith("Error"):
            st.success("PDF content loaded successfully!")
elif input_option == "DOCX":
    uploaded_docx = st.file_uploader("Upload DOCX File", type=["docx"])
    if uploaded_docx:
        input_text = handle_docx(uploaded_docx)
        if not input_text.startswith("Error"):
            st.success("DOCX content loaded successfully!")
elif input_option == "Audio":
    uploaded_audio = st.file_uploader("Upload Audio File (WAV only)", type=["wav"])
    if uploaded_audio:
        input_text = handle_audio(uploaded_audio)
        if not input_text.startswith("Error"):
            st.success("Audio transcribed successfully!")

if input_text and input_text.startswith("Error"):
    st.error(input_text)

word_count = st.slider("✂️ Desired summary length (words):", 20, 300, 100, step=10)
mode = st.radio("🧠 Choose summarization type:", ["Abstractive", "Extractive"], horizontal=True)
submitted = st.button("🔍 Summarize")

if submitted and input_text and not input_text.startswith("Error"):
    with st.spinner("Summarizing... please wait..."):
        result = process_text(
            input_text, 
            word_count, 
            mode="map_reduce" if mode == "Abstractive" else "stuff"
        )
        
        st.subheader("📜 Generated Summary")
        st.write(result)

        # Download button
        summary_bytes = io.BytesIO(result.encode("utf-8"))
        st.download_button(
            label="📂 Download Summary as TXT",
            data=summary_bytes,
            file_name="summary.txt",
            mime="text/plain"
        )
