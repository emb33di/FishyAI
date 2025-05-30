# Fix for torch.classes errors in StreamLit
import sys
try:
    if 'torch.classes' in sys.modules:
        sys.modules['torch.classes'].__path__ = []
except Exception:
    pass

# Continue with your existing imports
import streamlit as st
from agent import PropertyLawAgent
import time
import os
from dotenv import load_dotenv
import tempfile
import shutil
import json
import uuid

# Load environment variables from .env file (for local development)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="FishyAI - Property Law Assistant",
    page_icon="🐟",
    layout="wide"
)

# Custom CSS for a prettier interface
st.markdown("""
<style>
    /* Main Container Styling */
    .main {
        padding: 1.5rem;
    }
    
    /* Header Styling */
    .stTitle {
        color: #1e3a8a;
        margin-bottom: 0.5rem !important;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        padding: 0.75rem 0;
    }
    
    /* Custom message container with shadow */
    .message-container {
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #f3f4f6;
        border-left: 4px solid #6366f1;
    }
    
    /* Sources box styling */
    .sources-box {
        background-color: #f8fafc;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        border-left: 3px solid #0ea5e9;
        font-size: 0.9rem;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Description box */
    .description-box {
        background-color: #f0f9ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Response time indicator */
    .response-time {
        font-size: 0.8rem;
        color: #6b7280;
        text-align: right;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Create a unique user ID and store it in session state
if 'user_id' not in st.session_state:
    # Use browser cookie or session ID if available
    # For simplicity, we generate a UUID and store it in session state
    st.session_state.user_id = str(uuid.uuid4())

@st.cache_resource(show_spinner=False)
def initialize_agent(pdf_directory):
    """Cache the agent to prevent reinitialization on every refresh"""
    agent = PropertyLawAgent(pdf_directory)
    return agent

# Define the function to save chat history
def save_chat_history(user_id, chat_history):
    """Save chat history to file based on user ID"""
    os.makedirs("pdfs", exist_ok=True)
    history_file = os.path.join("pdfs", f"chat_history_{user_id}.json")
    try:
        with open(history_file, 'w') as f:
            json.dump(chat_history, f)
    except Exception as e:
        st.warning(f"Error saving chat history: {str(e)}")

# Initialize session state
if 'agent' not in st.session_state:
    try:
        with st.spinner("Initializing Property Law Assistant..."):
            st.session_state.agent = initialize_agent("pdfs")
            success, message = st.session_state.agent.load_pdfs()
            st.session_state.initial_load_message = message
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")

# Define function to load chat history
def load_chat_history(user_id):
    """Load chat history from file based on user ID"""
    os.makedirs("pdfs", exist_ok=True)
    history_file = os.path.join("pdfs", f"chat_history_{user_id}.json")
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.warning(f"Error loading chat history: {str(e)}")
        return []

# Load chat history from file instead of initializing empty
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history(st.session_state.user_id)

# Title and description
st.title("🐟 FishyAI - Your Property Law Assistant")
st.markdown("""
<div class="description-box">
    <p>This AI assistant can help you with questions about property law. Please <strong>ONLY</strong> use this tool as a guide for information synthesis. It is not meant to be a substitute for exam preparation.</p>
    <p> All context is drawn either from Professor Fisher's Property law website or from his slides. FishyAI does not own any of the content used as context. </p>
    <p>As instructed by Professor Fisher, please cite that you are using this tool/an AI if you use any of the material generated by it in your exam response. Have a great exam and Happy Fishing!</p>
</div>
""", unsafe_allow_html=True)

# Add sidebar with additional info
with st.sidebar:
    st.header("About FishyAI")
    st.markdown("Your AI assistant for property law questions")
    
    # Add a clear chat history button
    st.subheader("Chat Options")
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        # Also clear history for the agent
        st.session_state.agent.chat_history = []
        # Clear the saved chat history file
        save_chat_history(st.session_state.user_id, [])
        st.rerun()  # Rerun the app to reflect the cleared history

    # Add PDF info
    st.subheader("Loaded Documents")
    loaded_pdfs = st.session_state.agent.get_loaded_pdfs()
    if loaded_pdfs:
        for pdf in loaded_pdfs:
            st.markdown(f"- {pdf}")
    else:
        st.write("No documents loaded")


# Display initial loading message if exists
if 'initial_load_message' in st.session_state:
    st.info(st.session_state.initial_load_message)

# Update the function to display custom formatted messages
def display_message(role, content, sources=None, response_time=None, cost_info=None):
    if role == "user":
        message_class = "user-message"
        prefix = "🧑‍🎓 You"
    else:
        message_class = "assistant-message"
        prefix = "🐟 FishyAI"
    
    st.markdown(f"""
    <div class="message-container {message_class}">
        <strong>{prefix}:</strong><br/>
        {content}
    """, unsafe_allow_html=True)
    
    footer_elements = []
    
    if response_time is not None:
        footer_elements.append(f"Response time: {response_time:.2f} seconds")
    
    if cost_info is not None:
        tokens = cost_info.get("tokens", {})
        if tokens:
            total_tokens = tokens.get("total", tokens.get("input", 0) + tokens.get("output", 0))
            footer_elements.append(f"Tokens: {total_tokens}")
        
        query_cost = cost_info.get("cost", {}).get("query_cost")
        if query_cost is not None:
            footer_elements.append(f"Cost: ${query_cost:.6f}")
    
    if footer_elements:
        footer_html = " | ".join(footer_elements)
        st.markdown(f"""
        <div class="response-time">
            {footer_html}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Display chat history with custom formatting
for message in st.session_state.chat_history:
    display_message(
        role=message["role"], 
        content=message["content"],
        sources=message.get("sources"),
        response_time=message.get("response_time")
    )

# Chat input
if prompt := st.chat_input("Ask your question about property law..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.user_id, st.session_state.chat_history)
    
    # Display user message with custom formatting
    display_message(role="user", content=prompt)
    
    # Get AI response
    with st.spinner("FishyAI is thinking..."):
        start_time = time.time()
        result = st.session_state.agent.ask_question(prompt)
        response_time = time.time() - start_time
    
    # Extract cost information from result if available
    cost_info = None
    if "tokens" in result or "cost" in result:
        cost_info = {
            "tokens": result.get("tokens", {}),
            "cost": result.get("cost", {})
        }
    
    # Display assistant message with custom formatting
    display_message(
        role="assistant", 
        content=result["answer"],
        sources=result["sources"],
        response_time=response_time,
        cost_info=cost_info
    )
    
    # Add assistant response to chat history with cost info
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "response_time": response_time,
        "cost_info": cost_info
    })
    
    # Save the updated chat history
    save_chat_history(st.session_state.user_id, st.session_state.chat_history)