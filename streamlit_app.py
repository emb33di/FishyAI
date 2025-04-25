import streamlit as st
from agent import PropertyLawAgent
import time
import os
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables from .env file (for local development)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="FishyAI - Property Law Assistant",
    page_icon="🐟",
    layout="wide"
)

# Get API key from environment variable first, then fall back to Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("secrets", {}).get("OPENAI_API_KEY")

if not api_key:
    st.error("""
    OPENAI_API_KEY not found. Please set it as an environment variable:
    
    For local development:
    1. Create a .env file in the root directory
    2. Add: OPENAI_API_KEY=your_api_key_here
    
    For production:
    Set the OPENAI_API_KEY environment variable in your deployment platform
    """)
    st.stop()

# Initialize session state
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = PropertyLawAgent("pdfs")
        success, message = st.session_state.agent.load_pdfs()
        st.session_state.initial_load_message = message
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("🐟 FishyAI - Your Property Law Assistant")
st.markdown("""
This AI assistant can help you with questions about property law. 
Upload your PDF documents and ask questions to get detailed answers with sources.
""")

# File upload section
st.header("📚 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload your property law PDFs",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    # Create pdfs directory if it doesn't exist
    os.makedirs("pdfs", exist_ok=True)
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Process PDFs
    with st.spinner("Processing PDFs..."):
        success, message = st.session_state.agent.load_pdfs()
        if success:
            st.success(message)
        else:
            st.error(message)

# Display loaded PDFs
loaded_pdfs = st.session_state.agent.get_loaded_pdfs()
if loaded_pdfs:
    st.subheader("📖 Loaded Documents")
    for pdf in loaded_pdfs:
        st.write(f"- {pdf}")

# Chat interface
st.header("💬 Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask your question about property law"):
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