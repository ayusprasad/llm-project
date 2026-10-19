from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODELS = {
    "GPT OSS 20B": "openai/gpt-oss-20b",
    "GPT OSS 120B": "openai/gpt-oss-120b",
}


def build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    question: str,
    memory_turns: int,
) -> list[dict[str, str]]:
    """Build a bounded provider request from the current prompt and chat history."""
    prompt = system_prompt.strip() or "You are a helpful, concise assistant."
    messages = [{"role": "system", "content": prompt}]

    for turn in history[-max(0, memory_turns) :]:
        user_message = turn.get("user", "").strip()
        assistant_message = turn.get("assistant", "").strip()
        if user_message and assistant_message:
            messages.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            )

    current_question = question.strip()
    if current_question:
        messages.append({"role": "user", "content": current_question})
    return messages


def _render_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="Groq Chat", page_icon="💬")
    st.title("Groq Chat")
    st.caption("A focused demonstration of session-based chat with Groq.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Conversation settings")
        model_label = st.selectbox("Model", list(MODELS))
        memory_turns = st.slider("Recent turns to include", min_value=1, max_value=10, value=5)
        system_prompt = st.text_area(
            "System prompt",
            value="You are a helpful assistant. Give clear, practical answers.",
            max_chars=2_000,
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        _render_message(message)

    question = st.chat_input("Ask a question")
    if not question:
        return

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to your local .env file and restart Streamlit.")
        return

    _render_message({"role": "user", "content": question})
    try:
        request_messages = build_messages(
            system_prompt,
            st.session_state.chat_history,
            question,
            memory_turns,
        )
        with st.chat_message("assistant"):
            with st.spinner("Generating a response..."):
                response: Any = Groq(api_key=api_key).chat.completions.create(
                    model=MODELS[model_label],
                    messages=request_messages,
                )
            answer = response.choices[0].message.content or ""
            st.markdown(answer or "The model returned an empty response.")
    except Exception:
        st.error("The request failed. Check the API key, model access, and network connection, then retry.")
        return

    st.session_state.chat_history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )


if __name__ == "__main__":
    main()
