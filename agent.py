import os
import sys
import json
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from pdf_processor import PDFProcessor

class DocumentAgent:
    def __init__(self, pdf_directory: str = "pdfs"):
        self.pdf_directory = pdf_directory
        self.processor = PDFProcessor(pdf_directory)
        
    def process_documents(self) -> tuple[bool, str]:
        """Process all documents in the PDF directory."""
        return self.processor.load_and_process_pdfs()
    
    def query_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Query the processed documents for relevant information."""
        return self.processor.get_relevant_documents(query, k=k)
    
    def get_document_summary(self, query: str) -> str:
        """Get a summary of relevant documents for a query."""
        docs = self.query_documents(query)
        if not docs:
            return "No relevant documents found."
        
        summary = f"Found {len(docs)} relevant documents:\n\n"
        for i, doc in enumerate(docs, 1):
            summary += f"Document {i} (from {doc['source']}):\n"
            summary += f"{doc['content'][:500]}...\n\n"
        
        return summary

def main():
    st.title("Document Processing Agent")
    
    # Initialize agent
    agent = DocumentAgent()
    
    # Process documents
    with st.spinner("Processing documents..."):
        success, message = agent.process_documents()
        if success:
            st.success(message)
        else:
            st.error(message)
            return
    
    # Query interface
    st.subheader("Query Documents")
    query = st.text_input("Enter your query:")
    
    if query:
        with st.spinner("Searching documents..."):
            # Get document summary
            summary = agent.get_document_summary(query)
            st.write(summary)
            
            # Show detailed results
            st.subheader("Detailed Results")
            docs = agent.query_documents(query)
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document {i} (from {doc['source']})"):
                    st.write(doc['content'])

if __name__ == "__main__":
    main()