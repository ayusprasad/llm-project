]import langchain_groq
import streamlit as st
import os
from groq import Groq
import random
import nltk
import streamlit.components.v1 as components

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag, word_tokenize
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Function to get grammar-related information
def get_grammar_info(word):
    tokens = word_tokenize(word)
    pos_tags = pos_tag(tokens)
    return pos_tags

# Main function
def main():
    # Embed HubSpot script inside a div which will be in the body of the page
    components.html("""
    <div>
        <!-- Start of HubSpot Embed Code --> 
        <script type="text/javascript" id="hs-script-loader" src="//js.hs-scripts.com/47792726.js"></script> 
        <!-- End of HubSpot Embed Code -->
    </div>
    """, height=0, width=0)

    # Get Groq API key
    api_key = os.getenv("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")

    client = Groq(api_key="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")

    # Display the Groq logo
    spacer, col = st.columns([5, 1])

    # Streamlit UI components
    st.title("AI-Enhanced Vocabulary and Grammar Assistant!")
    st.write(
        "Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide grammar insights, or just chat. Discover, Learn, and Master New Words Effortlessly! Let's start our conversation.")

    # Sidebar for customization
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history",
                                            return_messages=True)

    user_question = st.text_input("Enter a word to get its grammar details:")

    # Manage chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key="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",
        model_name=model
    )

    if user_question:
        # Get grammar-related information
        grammar_info = get_grammar_info(user_question)

        # Construct prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),
                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),
            ]
        )

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # Generate chatbot response
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}

        # Append message to chat history and display responses
        grammar_message = f"Grammar Info for '{user_question}': {grammar_info}"
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)
        st.write(grammar_message)

    # Add a simple delay for demonstration purposes
    import time
    time.sleep(4)

    print("This message will display first.")
    print()  # Blank line
    time.sleep(1)

    print("Here's a message with a gap below it.")
    print("\n" * 3)  # Adds 3 blank lines
    time.sleep(1)

    print("This message will display after the gaps.")

# Entry point of the script
if __name__ == "__main__":
    main()
