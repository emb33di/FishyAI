import os
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

class PropertyLawAgent:
    def __init__(self, pdf_directory: str):
        # Load environment variables
        load_dotenv()
        
        # Get API key from Streamlit secrets or .env
        api_key = st.secrets.get("secrets", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or .env file")
        
        # Initialize OpenAI client with explicit configuration
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        
        self.pdf_directory = pdf_directory
        self.chat_history = []
        self.loaded_pdfs = []
        
    def load_pdfs(self) -> tuple[bool, str]:
        """Load and process all PDFs in the specified directory."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                return False, f"Created empty '{self.pdf_directory}' directory. Please add your PDF files there."
            
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                return False, f"No PDF files found in '{self.pdf_directory}' directory. Please add your PDF files."
            
            self.loaded_pdfs = pdf_files
            return True, f"Successfully loaded {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
        except Exception as e:
            return False, f"Error loading PDFs: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, str]:
        """Ask a question about property law and get an answer."""
        try:
            # Create a system message that includes context about the PDFs
            system_message = f"You are a property law assistant. You have access to the following documents: {', '.join(self.loaded_pdfs)}. Please provide accurate and helpful answers based on these documents."
            
            # Prepare messages for the chat
            messages = [
                {"role": "system", "content": system_message},
                *[{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history],
                {"role": "user", "content": question}
            ]
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "sources": self.loaded_pdfs
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def get_loaded_pdfs(self) -> list[str]:
        """Return list of currently loaded PDFs."""
        return self.loaded_pdfs
