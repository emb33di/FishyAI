import os
from typing import List, Dict
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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
        """Load and process all PDFs in the specified directory using OCR."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                return False, f"Created empty '{self.pdf_directory}' directory. Please add your PDF files there."
            
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                return False, f"No PDF files found in '{self.pdf_directory}' directory. Please add your PDF files."
            
            # Process each PDF using OCR-capable loader
            documents = []
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(self.pdf_directory, pdf_file)
                    st.info(f"Processing {pdf_file}...")
                    
                    # Use OCR-capable loader for all documents
                    loader = UnstructuredPDFLoader(
                        pdf_path, 
                        mode="elements", 
                        strategy="fast"
                    )
                    pdf_docs = loader.load()
                    st.info(f"{pdf_file}: {len(pdf_docs)} elements extracted")
                    
                    documents.extend(pdf_docs)
                    
                except Exception as e:
                    st.error(f"Error processing {pdf_file}: {str(e)}")
                    continue
            
            if not documents:
                return False, "No documents were successfully loaded from the PDFs."
            
            # Split documents into chunks
            st.info(f"Splitting {len(documents)} documents into chunks...")
            splits = self.text_splitter.split_documents(documents)
            st.info(f"Created {len(splits)} chunks")
            
            # Create vector store
            st.info("Creating vector store...")
            self.vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            return True, f"Successfully processed {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
        except Exception as e:
            return False, f"Error processing PDFs: {str(e)}"
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
        
        Returns:
            List[Dict]: List of relevant documents
        """
        if not self.vector_store:
            st.error("No vector store available")
            return []
        
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Return formatted documents
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