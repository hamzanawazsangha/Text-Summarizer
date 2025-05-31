import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import math
import io
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
import docx2txt
import PyPDF2
import os
from datetime import datetime
import speech_recognition as sr
from langdetect import detect
from googletrans import Translator

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
    st.caption("Built with 💡 Transformers, 🌐 Multilingual Support, and 🧠 Streamlit")

# --- Model Loaders ---
@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization", model="google/mt5-small")

@st.cache_resource
def load_extractive_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

abstractive_model = load_abstractive_model()
extractive_model = load_extractive_model()
translator = Translator()

# --- Text Chunking ---
def split_text(text, max_tokens=512):
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) < max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# --- Summarization Function ---
def summarize_text(text, word_limit, mode):
    if not text.strip():
        return "❗ Please enter some text."

    language = detect(text)
    if language != 'en':
        text = translator.translate(text, src=language, dest='en').text

    chunks = split_text(text)
    all_summaries = []

    max_length = math.ceil(word_limit * 1.5)
    min_length = max(10, math.floor(word_limit * 0.5))

    model = abstractive_model if mode == "Abstractive" else extractive_model

    for chunk in chunks:
        summary = model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        all_summaries.append(summary[0]['summary_text'])

    summary_text = ' '.join(all_summaries)

    if language != 'en':
        summary_text = translator.translate(summary_text, src='en', dest=language).text

    return summary_text

# --- Fetch YouTube Transcript ---
def get_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1].split("&")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x['text'] for x in transcript])
    except Exception as e:
        return f"Error: {e}"

# --- Fetch Article Text from URL ---
def get_text_from_url(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all('p')
        return " ".join([para.get_text() for para in paragraphs])
    except:
        return "Could not fetch or parse the article."

# --- Read PDF ---
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Read DOCX ---
def read_docx(file):
    return docx2txt.process(file)

# --- Read Audio File ---
def transcribe_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Error with speech recognition service."

# --- History Dashboard ---
history_dir = "summaries"
os.makedirs(history_dir, exist_ok=True)

st.sidebar.header("Upload Options")
input_option = st.sidebar.selectbox("Choose input type:", ["Text", "YouTube URL", "Article URL", "PDF", "DOCX", "Audio"])

input_text = ""

if input_option == "Text":
    input_text = st.text_area("📝 Enter text to summarize (any language):", height=300)
elif input_option == "YouTube URL":
    video_url = st.text_input("🎥 Enter YouTube Video URL:")
    if video_url:
        input_text = get_youtube_transcript(video_url)
elif input_option == "Article URL":
    article_url = st.text_input("📖 Enter Web Article URL:")
    if article_url:
        input_text = get_text_from_url(article_url)
elif input_option == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        input_text = read_pdf(uploaded_pdf)
elif input_option == "DOCX":
    uploaded_docx = st.file_uploader("Upload DOCX File", type=["docx"])
    if uploaded_docx:
        input_text = read_docx(uploaded_docx)
elif input_option == "Audio":
    uploaded_audio = st.file_uploader("Upload Audio File (WAV only)", type=["wav"])
    if uploaded_audio:
        input_text = transcribe_audio(uploaded_audio)

word_count = st.slider("✂️ Desired summary length (words):", 20, 300, 100, step=10)
mode = st.radio("🧠 Choose summarization type:", ["Abstractive", "Extractive"], horizontal=True)
submitted = st.button("🔍 Summarize")

if submitted and input_text:
    with st.spinner("Summarizing... please wait..."):
        result = summarize_text(input_text, word_count, mode)
        st.subheader("📜 Generated Summary")
        st.write(result)

        # Save to history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.txt"
        filepath = os.path.join(history_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result)

        # Download button
        summary_bytes = io.BytesIO(result.encode("utf-8"))
        st.download_button(
            label="📂 Download Summary as TXT",
            data=summary_bytes,
            file_name="summary.txt",
            mime="text/plain"
        )

# --- History Viewer ---
st.sidebar.header("📄 View Past Summaries")
history_files = sorted(os.listdir(history_dir), reverse=True)

if history_files:
    selected_file = st.sidebar.selectbox("Choose a file to view:", history_files)
    if selected_file:
        with open(os.path.join(history_dir, selected_file), "r", encoding="utf-8") as f:
            st.sidebar.markdown("---")
            st.sidebar.text_area("Summary Preview", f.read(), height=200)
else:
    st.sidebar.info("No saved summaries yet.")
