import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Change from Chroma to FAISS
import streamlit as st

class PDFProcessor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        
    def load_and_process_pdfs(self) -> tuple[bool, str]:
        """Load and process all PDFs in the specified directory."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                return False, f"Created empty '{self.pdf_directory}' directory. Please add your PDF files there."
            
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                return False, f"No PDF files found in '{self.pdf_directory}' directory. Please add your PDF files."
            
            # Process each PDF
            documents = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Create vector store using FAISS instead of Chroma
            self.vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            # Save the vector store to disk
            os.makedirs("faiss_index", exist_ok=True)
            self.vector_store.save_local("faiss_index")
            
            return True, f"Successfully processed {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
        except Exception as e:
            return False, f"Error processing PDFs: {str(e)}"
    
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if not self.vector_store:
            # Try to load from disk if it exists
            if os.path.exists("faiss_index"):
                try:
                    self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
                except Exception:
                    return []
            else:
                return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in docs
            ]
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []