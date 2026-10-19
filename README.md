# 📖 AI-Enhanced Vocabulary & Grammar Assistant

An interactive AI assistant built with **Streamlit**, **LangChain**, and **Groq** for rapid grammatical analysis, part-of-speech tagging (NLTK), and contextual vocabulary learning powered by LLaMA 3.3 and Mixtral.

---

## 🚀 Features
- **Ultra-Fast LLM Inference:** Powered by Groq's LPUs running LLaMA 3.3 and Mixtral models.
- **Automated POS Tagging:** Combines symbolic NLP (NLTK part-of-speech tagger) with LLM explanations for maximum precision.
- **Configurable Personas & Memory:** Adjust system prompts and conversation buffer length directly from the sidebar.
- **Secure Key Management:** Supports both environment variable `GROQ_API_KEY` and interactive user key inputs (keys are never logged or hardcoded).
- **Docker Ready:** Built-in multi-stage Docker container with pre-cached NLTK corpora for instant startup.

---

## 🛠️ Quickstart

### 1. Local Run
```bash
# Clone repository
git clone https://github.com/ayusprasad/llm-project.git
cd llm-project

# Install dependencies
pip install -r requirements.txt

# (Optional) Export Groq API key
export GROQ_API_KEY="your-groq-api-key"

# Launch Streamlit app
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 2. Run with Docker
```bash
# Build Docker image
docker build -t vocabulary-assistant .

# Run container (pass key as environment variable or enter in UI)
docker run -p 8501:8501 -e GROQ_API_KEY="your-key" vocabulary-assistant
```

---

## 📁 Project Structure
```text
llm-project/
├── app.py                     # Streamlit application entrypoint
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Production Docker configuration
├── .gitignore                 # Secrets and environment ignore rules
└── README.md                  # Project documentation
```
