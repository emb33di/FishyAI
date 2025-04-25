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
    # Model pricing in USD per 1K tokens (updated with actual prices)
    MODEL_PRICES = {
        "gpt-4.1-mini": {
            "input": 0.0004,     # $0.40/1M input tokens
            "cached_input": 0.0001,  # $0.10/1M cached input tokens (currently not used)
            "output": 0.0016     # $1.60/1M output tokens
        }
    }
    
    def __init__(self, pdf_directory: str):
        # Load environment variables
        load_dotenv()
        
        # Cost tracking
        self.cost_log_path = os.path.join(pdf_directory, "cost_tracking.json")
        self.total_cost = 0.0
        self.load_cost_data()
        
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

    def load_cost_data(self):
        """Load existing cost tracking data if available"""
        if os.path.exists(self.cost_log_path):
            try:
                with open(self.cost_log_path, 'r') as f:
                    cost_data = json.load(f)
                    self.total_cost = cost_data.get("total_cost", 0.0)
            except Exception as e:
                if 'streamlit' in sys.modules:
                    st.warning(f"Error loading cost data: {e}. Starting with zero.")
                else:
                    print(f"Error loading cost data: {e}. Starting with zero.")
    
    def save_cost_data(self):
        """Save cost tracking data to file"""
        try:
            os.makedirs(os.path.dirname(self.cost_log_path), exist_ok=True)
            with open(self.cost_log_path, 'w') as f:
                json.dump({
                    "total_cost": self.total_cost,
                    "last_updated": datetime.now().isoformat()
                }, f)
        except Exception as e:
            if 'streamlit' in sys.modules:
                st.warning(f"Error saving cost data: {e}")
            else:
                print(f"Error saving cost data: {e}")
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost based on tokens used and model pricing"""
        if model not in self.MODEL_PRICES:
            return 0.0  # Unknown model
            
        input_cost = (input_tokens / 1000) * self.MODEL_PRICES[model]["input"]
        output_cost = (output_tokens / 1000) * self.MODEL_PRICES[model]["output"]
        return input_cost + output_cost
        
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
            system_message = f"""You are a property law exam assistant. Use the following context to answer the questions about property law doctrine. 
            If the answer cannot be found in the context, say so. Always cite your sources and specifically the case from which the information is derived.
            
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
                
                # Track cost
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                query_cost = self.calculate_cost(model, input_tokens, output_tokens)
                self.total_cost += query_cost
                self.save_cost_data()
                
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
                },
                "cost": {
                    "query_cost": query_cost,
                    "total_cost": self.total_cost
                }
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "cost": {"query_cost": 0, "total_cost": self.total_cost}
            }
    
    def get_cost_summary(self):
        """Get a summary of the cost usage"""
        return {
            "total_cost": self.total_cost
        }
    
    def get_loaded_pdfs(self) -> list[str]:
        """Return list of currently loaded PDFs."""
        if not os.path.exists(self.pdf_directory):
            return []
        return [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]