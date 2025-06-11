# 📄 Universal Text Summarizer

A simple web app that summarizes long texts using AI. Just paste your text, click a button, and get a concise summary instantly.

## 🌟 Features

- Summarizes any text up to 3,000 words
- Works fast on regular computers (no fancy GPU needed)
- Clean, simple interface anyone can use
- Powered by advanced AI (LED-Large model)

## 🚀 How to Use

1. **Paste your text** in the big text box
2. Click the **"Summarize"** button
3. Get your summary in seconds!

> 💡 Tip: The app automatically shortens very long texts to ~3,000 words if needed.

## 🛠️ For Developers

### Requirements
- Python 3.7+
- Streamlit
- PyTorch
- HuggingFace Transformers
- LangChain

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the App
```bash
streamlit run app.py
```

## ⚙️ Technical Details
- Uses the `allenai/led-large-16384` model (good for long documents)
- Runs on CPU (no special hardware needed)
- Built with Streamlit for easy web access

## 🤝 Contributing
Found a bug? Want to improve something? Feel free to:
- Open an issue
- Submit a pull request

## 📜 License
MIT License - free to use and modify

---
