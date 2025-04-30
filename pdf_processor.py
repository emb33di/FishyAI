import os
import pickle
import hashlib
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Global cache for embeddings model
@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_embeddings():
    """Create a cached instance of OpenAI embeddings to reuse across sessions"""
    return OpenAIEmbeddings()

@st.cache_resource
def load_vectorstore_from_disk(_embedding_model):
    """Load the vector store from disk once and cache it globally"""
    if os.path.exists("faiss_index"):
        try:
            return FAISS.load_local("faiss_index", _embedding_model)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
    return None

@st.cache_resource
def create_and_save_vectorstore(_documents, _embedding_model):
    """Create vector store from documents and cache it"""
    vector_store = FAISS.from_documents(
        documents=_documents,
        embedding=_embedding_model
    )
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    return vector_store

class PDFProcessor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        # Use the cached embedding model
        self.embeddings = get_cached_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for slides
            chunk_overlap=100,  # Less overlap
            length_function=len,
        )
        self.vector_store = load_vectorstore_from_disk(self.embeddings)
        
        # Cache-related paths
        self.cache_dir = os.path.join(pdf_directory, ".cache")
        self.metadata_path = os.path.join(self.cache_dir, "cache_metadata.pkl")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load cache metadata if it exists
        self.cache_metadata = self._load_cache_metadata()
        
    def _load_cache_metadata(self):
        """Load cache metadata or create new if not exists"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {"pdfs": {}}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk"""
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.cache_metadata, f)
    
    def _get_pdf_hash(self, pdf_path):
        """Calculate hash of PDF file for change detection"""
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _is_cache_valid(self):
        """Check if cache is valid by comparing file hashes"""
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        # Check if cached files match current files
        cached_pdfs = set(self.cache_metadata["pdfs"].keys())
        current_pdfs = set(pdf_files)
        
        if cached_pdfs != current_pdfs:
            return False
        
        # Check if any PDF has been modified
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            current_hash = self._get_pdf_hash(pdf_path)
            cached_hash = self.cache_metadata["pdfs"].get(pdf_file)
            
            if current_hash != cached_hash:
                return False
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            return False
            
        return True
        
    def load_and_process_pdfs(self) -> tuple[bool, str]:
        """Load and process all PDFs in the specified directory, using cache when possible."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                return False, f"Created empty '{self.pdf_directory}' directory. Please add your PDF files there."
            
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                return False, f"No PDF files found in '{self.pdf_directory}' directory. Please add your PDF files."
            
            # Check if we can use cache
            if self._is_cache_valid():
                self.vector_store = load_vectorstore_from_disk(self.embeddings)
                if self.vector_store:
                    return True, f"Using cached data for {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
            # Process each PDF
            documents = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                
                # Update cache metadata with current file hash
                self.cache_metadata["pdfs"][pdf_file] = self._get_pdf_hash(pdf_path)
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Create and save vector store (cached)
            self.vector_store = create_and_save_vectorstore(splits, self.embeddings)
            
            # Save updated cache metadata
            self._save_cache_metadata()
            
            return True, f"Successfully processed {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
        except Exception as e:
            return False, f"Error processing PDFs: {str(e)}"
    
    def get_relevant_documents(self, query: str, k_cases: int = 2, k_slides: int = 2, k_general: int = 2) -> List[Dict]:
        """
        Retrieve relevant documents for a query, separating cases, slides, and general readings.
        
        Args:
            query (str): The search query
            k_cases (int): Number of case documents to retrieve
            k_slides (int): Number of slide documents to retrieve
            k_general (int): Number of general reading documents to retrieve
        
        Returns:
            List[Dict]: Combined list of relevant documents
        """
        if not self.vector_store:
            # Use cached loading
            self.vector_store = load_vectorstore_from_disk(self.embeddings)
            if not self.vector_store:
                return []
        
        try:
            # Search all documents
            total_k = k_cases + k_slides + k_general
            all_docs = self.vector_store.similarity_search(query, k=total_k)
            
            # Categorize documents
            slides_docs = []
            cases_docs = []
            general_docs = []
            
            for doc in all_docs:
                source = doc.metadata.get("source", "Unknown")
                source_lower = source.lower()
                
                # Check document type
                if " v. " in source or " v " in source:  # Case
                    cases_docs.append(doc)
                elif "slides" in source_lower or "Slides" in source:  # Slides
                    slides_docs.append(doc)
                else:  # General readings
                    general_docs.append(doc)
            
            # Additional targeted searches if needed
            # If we don't have enough slides
            if len(slides_docs) < k_slides:
                try:
                    # Try to get more slides specifically
                    slides_filter = lambda doc: "slides" in doc.metadata.get("source", "").lower() or "Slides" in doc.metadata.get("source", "")
                    more_slides = self.vector_store.similarity_search(
                        query, 
                        k=k_slides - len(slides_docs), 
                        filter=slides_filter
                    )
                    # Add unique slides
                    for slide in more_slides:
                        if slide.page_content not in [d.page_content for d in slides_docs]:
                            slides_docs.append(slide)
                except:
                    pass  # If filtering fails, continue with what we have
            
            # If we don't have enough cases
            if len(cases_docs) < k_cases:
                try:
                    # Try to get more cases specifically
                    cases_filter = lambda doc: (" v. " in doc.metadata.get("source", "") or 
                                              " v " in doc.metadata.get("source", ""))
                    more_cases = self.vector_store.similarity_search(
                        query, 
                        k=k_cases - len(cases_docs), 
                        filter=cases_filter
                    )
                    # Add unique cases
                    for case in more_cases:
                        if case.page_content not in [d.page_content for d in cases_docs]:
                            cases_docs.append(case)
                except:
                    pass  # If filtering fails, continue with what we have
            
            # If we don't have enough general readings
            if len(general_docs) < k_general:
                try:
                    # Try to get more general readings
                    general_filter = lambda doc: ("slides" not in doc.metadata.get("source", "").lower() and
                                               "Slides" not in doc.metadata.get("source", "") and
                                               " v. " not in doc.metadata.get("source", "") and
                                               " v " not in doc.metadata.get("source", ""))
                    more_general = self.vector_store.similarity_search(
                        query, 
                        k=k_general - len(general_docs), 
                        filter=general_filter
                    )
                    # Add unique general readings
                    for gen_doc in more_general:
                        if gen_doc.page_content not in [d.page_content for d in general_docs]:
                            general_docs.append(gen_doc)
                except:
                    pass  # If filtering fails, continue with what we have
            
            # Trim to desired counts
            slides_docs = slides_docs[:k_slides]
            cases_docs = cases_docs[:k_cases]
            general_docs = general_docs[:k_general]
            
            # Combine all documents, prioritizing slides, then cases, then general readings
            combined_docs = slides_docs + cases_docs + general_docs
            
            # Determine document type for each document
            def get_document_type(source):
                if " v. " in source or " v " in source:
                    return "case"
                elif "slides" in source.lower() or "Slides" in source:
                    return "slide"
                else:
                    return "general"
            
            # Return formatted documents
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": get_document_type(doc.metadata.get("source", "Unknown"))
                }
                for doc in combined_docs
            ]
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []