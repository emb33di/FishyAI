import streamlit as st
from agent import PropertyLawAgent
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="FishyAI - Property Law Assistant",
    page_icon="🐟",
    layout="wide"
)

# Get API key from either Streamlit secrets or .env file
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Please set it in Streamlit secrets or .env file.")
    st.stop()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = PropertyLawAgent("pdfs")
    success, message = st.session_state.agent.load_pdfs()
    st.session_state.initial_load_message = message

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("🐟 FishyAI - Your Property Law Assistant")
st.markdown("""
This AI assistant can help you with questions about property law in health data. 
The assistant has been pre-loaded with relevant property law documents and can provide detailed answers with sources.
""")

# Display available PDFs
st.sidebar.header("Available Documents")
pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
for pdf in pdf_files:
    st.sidebar.markdown(f"- {pdf}")

# Display initial load message
if 'initial_load_message' in st.session_state:
    st.info(st.session_state.initial_load_message)

# Chat interface
st.header("Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask your question about property law in health data"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            result = st.session_state.agent.ask_question(prompt)
            end_time = time.time()
            
            # Display answer
            st.write(result["answer"])
            
            # Display sources if available
            if result["sources"]:
                st.markdown("**Sources:**")
                for source in result["sources"]:
                    st.markdown(f"- {source}")
            
            # Display response time
            st.caption(f"Response time: {end_time - start_time:.2f} seconds")
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    }) 