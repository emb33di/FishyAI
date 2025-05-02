import os
import sys
import json
from typing import Dict
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from pdf_processor import DocumentProcessor

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
        self.pdf_processor = DocumentProcessor(pdf_directory)
        self.chat_history = []
        
    @st.cache_data(show_spinner=False)
    def load_pdfs(_self) -> tuple[bool, str]:
        """Load and process all PDFs in the specified directory."""
        return _self.pdf_processor.process_documents()
    
    def ask_question(self, question: str) -> Dict[str, any]:
        """Ask a question about property law and get an answer."""
        try:
            # Get relevant documents for the question
            relevant_docs = self.pdf_processor.query(  # Changed from get_relevant_documents()
                question, 
                k=7  # Use a single k parameter instead of multiple ones
            )
            
            # Sort relevant_docs to prioritize slides
            relevant_docs.sort(key=lambda doc: 0 if "Slides" in doc['source'].lower() else 1)
            
            # Create combined context without separating by type
            context = ""
            for doc in relevant_docs:
                # Make it clearer that this content is directly from the file
                content = f"SOURCE: {doc['source']}\nCONTENT FROM THIS FILE:\n{doc['content']}\n"
                context += content + "\n---\n"
            
            # Create a system message that emphasizes checking all types of content
            system_message = f"""You are a property law exam assistant. Use the following context to answer questions about property law doctrine.

            1. For each statement you make, explicitly cite the source from the context provided.
            2. Use the format: (Source) after each citation listing file name as source.
            3. Always include verbatim quotes from the pdf if citing a legal case. 
            4. Each source section begins with "SOURCE:" followed by the filename and then "CONTENT FROM THIS FILE:".
            5. Only cite sources that are actually provided in the context.
            6. If you go beyond provided context, explicitly state "I'm relying on outside context for this information" in your answer.
            7. Synthesize a comprehensive answer covering all relevant material.

            Context:
            {context}
            """
            
            # Before API call, calculate approximate tokens and trim history if needed
            history_messages = []
            max_history_tokens = 2000  # Set your budget

            # Add messages until we hit the budget (rough estimation)
            for msg in reversed(self.chat_history):
                # Estimate tokens (very roughly)
                estimated_tokens = len(msg["content"]) / 4  # ~4 chars per token on average
                if estimated_tokens > max_history_tokens:
                    break
                max_history_tokens -= estimated_tokens
                history_messages.insert(0, msg)  # Add at beginning to maintain order

            messages = [
                {"role": "system", "content": system_message},
                *[{"role": msg["role"], "content": msg["content"]} for msg in history_messages],
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
            
            # Extract only the sources mentioned in the answer
            used_sources = []
            for doc in relevant_docs:
                # Remove 'pdf/' prefix from source name for cleaner citation
                source_name = doc['source']
                clean_source_name = source_name.replace('pdf/', '')
                
                # Check if either the full path or clean name is in the answer
                if source_name in answer or clean_source_name in answer:
                    # Always add the clean name to sources
                    used_sources.append(clean_source_name)

            # Remove duplicates while preserving order
            used_sources = list(dict.fromkeys(used_sources))
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Limit chat history to the last 4 turns (8 messages)
            if len(self.chat_history) > 8:
                self.chat_history = self.chat_history[-8:]
            
            return {
                "answer": answer,
                "sources": used_sources,  # Include clean source names without pdf/ prefix
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