import streamlit as st
import os
import time
from agent import PropertyLawAgent

# Page configuration
st.set_page_config(
    page_title="FishyAI - Property Law Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for a prettier interface
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #e0f7fa;
        border-bottom-right-radius: 0.2rem;
    }
    .chat-message.assistant {
        background-color: #f3f4f6;
        border-bottom-left-radius: 0.2rem;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .stTextInput {
        padding-bottom: 2rem;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
        border-left: 4px solid #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

def display_message(role, content):
    """Display a chat message with the appropriate styling."""
    if role == "user":
        avatar = "👤"
        message_class = "user"
    else:
        avatar = "🤖"
        message_class = "assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<div class='main-header'><h1>⚖️ FishyAI - Property Law Assistant</h1></div>", unsafe_allow_html=True)
    
    # Initialize session state
    if "agent" not in st.session_state:
        # Get PDF directory - adjust this path as needed
        pdf_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
        st.session_state.agent = PropertyLawAgent(pdf_directory)
        st.session_state.chat_history = []
        st.session_state.pdfs_loaded = False
    
    # Sidebar for PDF management
    with st.sidebar:
        st.header("PDF Documents")
        
        if not st.session_state.pdfs_loaded:
            if st.button("Load PDF Documents"):
                with st.spinner("Loading and processing PDFs..."):
                    success, message = st.session_state.agent.load_pdfs()
                    if success:
                        st.success(message)
                        st.session_state.pdfs_loaded = True
                    else:
                        st.error(message)
        else:
            st.success("PDFs loaded successfully!")
            
        # Display loaded PDFs
        loaded_pdfs = st.session_state.agent.get_loaded_pdfs()
        if loaded_pdfs:
            st.write("Loaded documents:")
            for pdf in loaded_pdfs:
                st.write(f"- {pdf}")
        
        # Reset chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.agent.chat_history = []
            st.experimental_rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_message(message["role"], message["content"])
        if "sources" in message and message["sources"]:
            st.markdown(
                f"""<div class="source-box">
                <strong>Sources:</strong> {', '.join(message["sources"])}
                </div>""",
                unsafe_allow_html=True
            )
    
    # Chat input
    with st.container():
        user_input = st.text_input("Ask a question about property law:", key="user_input")
        
        if user_input:
            # Display user message
            display_message("user", user_input)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response from agent
            with st.spinner("Thinking..."):
                if not st.session_state.pdfs_loaded:
                    time.sleep(1)  # Simulate thinking
                    answer = "Please load the PDF documents first using the button in the sidebar."
                    sources = []
                else:
                    response = st.session_state.agent.ask_question(user_input)
                    answer = response["answer"]
                    sources = response["sources"]
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
            # Clear the input box and rerun to update the display
            st.session_state.user_input = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()
