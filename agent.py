import os
import sys
import json
from typing import Dict
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from pdf_processor import PDFProcessor

class PropertyLawAgent:    
    def __init__(self, pdf_directory: str):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable first, then fall back to Streamlit secrets
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or Streamlit secrets")
        
        # Validate API key format
        if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
            raise ValueError("Invalid API key format. API key should start with 'sk-' or 'sk-proj-'")
        
        # Initialize OpenAI client with error handling
        try:
            self.client = OpenAI(api_key=api_key)  # This disables any custom HTTP client
            # Test the client with a simple request - but don't output to streamlit in CLI mode
            self.client.models.list()
            if 'streamlit' in sys.modules:
                st.write("OpenAI client initialized and tested successfully")
        except Exception as e:
            if 'streamlit' in sys.modules:
                st.error(f"Error initializing OpenAI client: {str(e)}")
            else:
                print(f"Error initializing OpenAI client: {str(e)}")
            raise
        
        self.pdf_directory = pdf_directory
        self.pdf_processor = PDFProcessor(pdf_directory)
        self.chat_history = []
        
    def load_pdfs(self) -> tuple[bool, str]:
        """Load and process all PDFs in the specified directory."""
        return self.pdf_processor.load_and_process_pdfs()
    
    def ask_question(self, question: str) -> Dict[str, any]:
        """Ask a question about property law and get an answer."""
        try:
            # Get relevant documents for the question
            relevant_docs = self.pdf_processor.get_relevant_documents(question)
            
            # Create context from relevant documents
            context = "\n\n".join([f"From {doc['source']}:\n{doc['content']}" for doc in relevant_docs])
            
            # Create a system message that includes context
            system_message = f"""You are a property law exam assistant. Use the following context that includes pdfs of cases and classroom teaching slides to answer the questions about property law doctrine. 
            If the answer cannot be found in the context, say so. From each pdf extract the legal case names and cite the cases used at the end of a sentence in parenthesis. 
            Look through all relevant sources covering the question, and synthesize a brief effective answer in a organized manner.
            
            Context:
            {context}
            """
            
            # Prepare messages for the chat
            messages = [
                {"role": "system", "content": system_message},
                *[{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history],
                {"role": "user", "content": question}
            ]
            
            model = "gpt-4.1-mini" 
            
            # Get response from OpenAI
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            except Exception as e:
                if 'streamlit' in sys.modules:
                    st.error(f"Error calling OpenAI API: {str(e)}")
                else:
                    print(f"Error calling OpenAI API: {str(e)}")
                raise
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "sources": [doc["source"] for doc in relevant_docs],
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def get_loaded_pdfs(self) -> list[str]:
        """Return list of currently loaded PDFs."""
        if not os.path.exists(self.pdf_directory):
            return []
        return [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]