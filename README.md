# Groq Chat — A Focused AI Conversation Demo

[![CI](https://github.com/ayusprasad/llm-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/llm-project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Inference-Groq-F55036)

A small Streamlit chat application that sends a bounded conversation history to Groq and keeps credentials outside the source tree.

## What it demonstrates

- A responsive chat interface with session-scoped history.
- A configurable system prompt, model, and history window.
- Clear handling when the API key is missing or the provider request fails.
- A small, testable message builder separated from the Streamlit interaction.

## Architecture

```text
Browser
  └─ Streamlit UI (app.py)
       ├─ Session state: recent user and assistant messages
       └─ Groq Chat Completions API
```

Conversation history lives in the current Streamlit session and is not written to a database.

## Tech stack

Python 3.11+, Streamlit, Groq Python SDK, pytest, Ruff, GitHub Actions.

## Run locally

### Prerequisites

- Python 3.11 or newer
- A Groq API key

### Install

```bash
git clone https://github.com/ayusprasad/llm-project.git
cd llm-project
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and set `GROQ_API_KEY` to your own key. The local `.env` file is ignored by Git.

### Start

```bash
streamlit run app.py
```

Streamlit prints the local URL when the server starts.

## Quality checks

```bash
ruff check app.py tests
pytest -q
python -m compileall -q app.py tests
```

GitHub Actions runs the same lint, test, and compile checks on pushes and pull requests.

## Security

Never commit API keys. If a key was committed previously, remove it from use and rotate it with the provider; deleting it from the current source does not remove it from Git history.
