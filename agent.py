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
            # Get relevant documents for the question with all three types
            relevant_docs = self.pdf_processor.get_relevant_documents(
                question, 
                k_cases=2, 
                k_slides=2,
                k_general=2
            )
            
            # Separate documents by type for the context
            slides_context = ""
            cases_context = ""
            general_context = ""
            
            for doc in relevant_docs:
                doc_type = doc.get("type", "unknown")
                content = f"From {doc['source']}:\n{doc['content']}"
                
                if doc_type == "slide":
                    slides_context += content + "\n\n"
                elif doc_type == "case":
                    cases_context += content + "\n\n"
                else:  # general
                    general_context += content + "\n\n"
            
            # Create a combined context with clear sections
            context = ""
            if slides_context:
                context += "SLIDE CONTENT:\n" + slides_context
            if cases_context:
                context += "CASE CONTENT:\n" + cases_context
            if general_context:
                context += "GENERAL READING CONTENT:\n" + general_context
            
            # Create a system message that emphasizes checking all types of content
            system_message = f"""You are a property law exam assistant. Use the following context that includes cases, classroom slides, and general readings to answer questions about property law doctrine.

IMPORTANT INSTRUCTIONS:
1. For each statement you make, explicitly cite the source from the context provided.
2. Use the format: (Source: filename.pdf) after each citation.
3. Look through ALL available content - slides, cases, and general readings.
4. Slides typically contain key points from the professor, so prioritize information from slides when available.
5. Cases provide legal precedents and reasoning, so make sure to reference relevant cases.
6. General readings provide additional context and explanation.
7. Only cite sources that are actually provided in the context below.
8. If you go beyond provided context, say so explicitly.
9. Synthesize a comprehensive answer covering all relevant material from the available sources.

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
            
            # Extract actual sources mentioned in the answer
            mentioned_sources = []
            for doc in relevant_docs:
                source_name = os.path.basename(doc['source'])
                if source_name in answer:
                    mentioned_sources.append(doc['source'])
            
            # If no sources mentioned but we have relevant docs, use those
            if not mentioned_sources and relevant_docs:
                mentioned_sources = [doc["source"] for doc in relevant_docs]
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "sources": mentioned_sources,  # Use verified sources
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