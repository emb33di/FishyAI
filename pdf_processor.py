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
                try:
                    pdf_path = os.path.join(self.pdf_directory, pdf_file)
                    st.write(f"Processing {pdf_file}...")  # Debug log
                    loader = PyPDFLoader(pdf_path)
                    pdf_docs = loader.load()
                    st.write(f"Loaded {len(pdf_docs)} pages from {pdf_file}")  # Debug log
                    documents.extend(pdf_docs)
                    
                    # Update cache metadata with current file hash
                    self.cache_metadata["pdfs"][pdf_file] = self._get_pdf_hash(pdf_path)
                except Exception as e:
                    st.error(f"Error processing {pdf_file}: {str(e)}")  # Debug log
                    continue
            
            if not documents:
                return False, "No documents were successfully loaded from the PDFs."
            
            # Split documents into chunks
            st.write(f"Splitting {len(documents)} documents into chunks...")  # Debug log
            splits = self.text_splitter.split_documents(documents)
            st.write(f"Created {len(splits)} chunks")  # Debug log
            
            # Create and save vector store (cached)
            st.write("Creating vector store...")  # Debug log
            self.vector_store = create_and_save_vectorstore(splits, self.embeddings)
            
            # Save updated cache metadata
            self._save_cache_metadata()
            
            return True, f"Successfully processed {len(pdf_files)} PDFs: {', '.join(pdf_files)}"
            
        except Exception as e:
            return False, f"Error processing PDFs: {str(e)}"
    
    def get_relevant_documents(self, query: str, k: int = 6) -> List[Dict]:
        """
        Retrieve relevant documents for a query, returning the most semantically relevant documents regardless of type.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
        
        Returns:
            List[Dict]: List of relevant documents
        """
        if not self.vector_store:
            # Use cached loading
            st.write("Loading vector store from disk...")  # Debug log
            self.vector_store = load_vectorstore_from_disk(self.embeddings)
            if not self.vector_store:
                st.error("No vector store available")  # Debug log
                return []
        
        try:
            # Search all documents
            st.write(f"Searching for documents with query: {query}")  # Debug log
            docs = self.vector_store.similarity_search(query, k=k)
            st.write(f"Found {len(docs)} relevant documents")  # Debug log
            
            # Log the sources of found documents
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            st.write(f"Document sources: {sources}")  # Debug log
            
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