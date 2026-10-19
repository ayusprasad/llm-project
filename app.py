import os
import nltk
import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

st.set_page_config(
    page_title="AI Vocabulary & Grammar Assistant",
    page_icon="📖",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def ensure_nltk_corpora():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


ensure_nltk_corpora()
from nltk import pos_tag, word_tokenize


def get_grammar_info(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    return tags


def main():
    st.title("📖 AI-Enhanced Vocabulary & Grammar Assistant")
    st.markdown(
        "Explore words, analyze grammatical structures, parts of speech, and practice contextual vocabulary powered by **Groq** high-speed inference."
    )

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # API Key retrieval
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at https://console.groq.com"
        )
        if not api_key:
            st.info("👋 Welcome! Please enter your **Groq API Key** in the sidebar to activate the assistant.")

    system_prompt = st.sidebar.text_area(
        "System Persona",
        value="You are an expert linguistics tutor. Explain vocabulary meanings, nuance, origin, and proper grammar clearly with illustrative sentences.",
        height=100
    )

    model_choice = st.sidebar.selectbox(
        "Groq LLM Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    )

    memory_len = st.sidebar.slider("Conversation Memory (turns)", 1, 10, value=5)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Enter a word, phrase, or sentence to analyze:")

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "pos_tags" in msg:
                st.caption(f"**Parts of Speech Tags:** `{msg['pos_tags']}`")

    if user_query:
        if not api_key:
            st.error("Please provide a Groq API Key in the sidebar.")
            return

        pos_tags = get_grammar_info(user_query)

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing with Groq..."):
                try:
                    groq_chat = ChatGroq(
                        groq_api_key=api_key,
                        model_name=model_choice
                    )

                    messages = [
                        SystemMessage(content=system_prompt)
                    ]
                    # Append recent conversation history
                    recent = st.session_state.chat_history[-memory_len*2:]
                    for m in recent[:-1]:
                        if m["role"] == "user":
                            messages.append(HumanMessagePromptTemplate.from_template("{text}").format(text=m["content"]))
                        else:
                            messages.append(SystemMessage(content=m["content"]))

                    prompt = f"Analyze the following word or sentence. Detail its part of speech, definition, grammatical function, and give 2 clear examples:\n\n'{user_query}'"
                    response = groq_chat.invoke(prompt)
                    response_text = response.content

                    st.markdown(response_text)
                    st.caption(f"**Parts of Speech Tags:** `{pos_tags}`")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "pos_tags": pos_tags
                    })
                except Exception as e:
                    st.error(f"Error communicating with Groq API: {e}")


if __name__ == "__main__":
    main()
