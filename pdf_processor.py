import os
import pickle
import hashlib
from typing import List, Dict, Tuple
from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from datetime import datetime


def _hash_file(path: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


@st.cache_resource(show_spinner=False)
def get_openai_embeddings():
    """Global cached embeddings to prevent recreating on instance creation"""
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        raise RuntimeError("Failed to initialize OpenAI embeddings")


class DocumentProcessor:
    """
    Extracts text from PDFs and PPTXs, splits into chunks,
    and indexes content using OpenAI embeddings with cost-saving caching.
    """
    def __init__(
        self,
        docs_dir: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        cache_subdir: str = ".cache"
    ):
        self.docs_dir = docs_dir
        self.cache_dir = os.path.join(docs_dir, cache_subdir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Get embeddings model - cached to prevent multiple initializations
        self.embeddings = get_openai_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.index = self._load_index()
        self.meta_path = os.path.join(self.cache_dir, 'metadata.pkl')
        self.metadata = self._load_metadata()
        self.chunks_cache_path = os.path.join(self.cache_dir, 'chunks.pkl')

    @st.cache_resource(show_spinner=False)
    def _load_index(self):
        """Load the index with error handling"""
        try:
            index_path = os.path.join(self.cache_dir, 'faiss_index')
            if os.path.exists(index_path):
                try:
                    return FAISS.load_local(index_path, self.embeddings)
                except Exception as e:
                    print(f"Error loading index: {e}")
                    # If embedding dimension changed or other error, we need to reindex
                    import shutil
                    shutil.rmtree(index_path)
            return None
        except Exception as e:
            print(f"Unexpected error in _load_index: {e}")
            return None

    def _load_metadata(self) -> Dict[str, str]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save_metadata(self):
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
    def _load_cached_chunks(self) -> List[Document]:
        """Load cached chunks to avoid regenerating embeddings"""
        if os.path.exists(self.chunks_cache_path):
            try:
                with open(self.chunks_cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached chunks: {e}")
        return None
        
    def _save_cached_chunks(self, chunks: List[Document]):
        """Save chunks to cache to avoid regenerating embeddings"""
        with open(self.chunks_cache_path, 'wb') as f:
            pickle.dump(chunks, f)

    def _needs_reindex(self, files: List[str]) -> bool:
        """Determine if reindexing is required based on file changes or missing index"""
        # Check if index exists
        if not self.index:
            print("Index not found, reindexing required")
            return True
            
        # Check if the number of files changed
        indexed_files = set(self.metadata.keys())
        current_files = set(files)
        
        if indexed_files != current_files:
            print(f"File list changed. Previously had {len(indexed_files)}, now have {len(current_files)}")
            return True
            
        # Check if any file content has changed
        for file in files:
            path = os.path.join(self.docs_dir, file)
            current_hash = _hash_file(path)
            if self.metadata.get(file) != current_hash:
                print(f"File {file} changed, reindexing required")
                return True
                
        print("No changes detected, using existing index")
        return False

    def _extract_text(self, path: str) -> List[Document]:
        """Extract text from PDFs or PPTXs based on file extension"""
        try:
            print(f"Extracting text from: {path}")
            
            # Determine file type and use appropriate loader
            if path.lower().endswith('.pdf'):
                docs = PyPDFLoader(path).load()
            elif path.lower().endswith(('.ppt', '.pptx')):
                docs = UnstructuredPowerPointLoader(path).load()
            else:
                print(f"Unsupported file type: {path}")
                return []
                
            if any(d.page_content.strip() for d in docs):
                print(f"Successfully extracted {len(docs)} pages from {path}")
                # Clean up the extracted text
                for doc in docs:
                    content = doc.page_content
                    # Replace multiple spaces with single space
                    content = ' '.join(content.split())
                    # Additional cleanup for presentation files
                    if path.lower().endswith(('.ppt', '.pptx')):
                        content = content.replace('•', '- ')  # Fix bullet points
                    doc.page_content = content
                return docs
                
            print(f"No content extracted from {path}")
            return []
            
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")
            return []

    def _create_index(self, chunks):
        """Create FAISS index with OpenAI embeddings"""
        print(f"Creating index with OpenAI embeddings for {len(chunks)} chunks...")
        return FAISS.from_documents(chunks, self.embeddings)

    @st.cache_data(show_spinner=False) 
    def process_documents(self) -> Tuple[bool, str]:
        """Load, split, and index all documents, using cache when valid to minimize API calls."""
        if not os.path.isdir(self.docs_dir):
            return False, f"Directory not found: {self.docs_dir}"

        # Get all PDF and PPTX files
        files = [f for f in os.listdir(self.docs_dir) if f.lower().endswith(('.pdf', '.ppt', '.pptx'))]
        if not files:
            return False, "No PDF or PPTX files found."
            
        # Check last processing timestamp to prevent frequent reprocessing
        timestamp_file = os.path.join(self.cache_dir, 'last_processed.txt')
        
        # Add a hash of the files to check if anything changed
        files_hash_file = os.path.join(self.cache_dir, 'files_hash.txt')
        current_files_hash = hashlib.md5(str(sorted(files)).encode()).hexdigest()
        
        # If hash exists and matches, use existing index
        if os.path.exists(files_hash_file) and os.path.exists(os.path.join(self.cache_dir, 'faiss_index')):
            with open(files_hash_file, 'r') as f:
                saved_hash = f.read().strip()
                if saved_hash == current_files_hash:
                    print("Documents collection unchanged, using existing index")
                    with open(timestamp_file, 'r') as f:
                        last_processed = f.read().strip()
                    return True, f"Using cached index from {last_processed} for {len(files)} documents."
        
        # If index exists and no files have changed, use cached index
        if os.path.exists(timestamp_file) and not self._needs_reindex(files):
            with open(timestamp_file, 'r') as f:
                last_processed = f.read().strip()
            return True, f"Using cached index from {last_processed} for {len(files)} documents."

        # Check if we have cached chunks that we can reuse
        cached_chunks = self._load_cached_chunks()
        if cached_chunks and all(file in self.metadata for file in files):
            # Only use cached chunks if we have all files accounted for
            chunks = cached_chunks
            print(f"Using {len(chunks)} cached chunks to avoid regenerating embeddings")
        else:
            # Process files from scratch
            all_docs: List[Document] = []
            for file in files:
                path = os.path.join(self.docs_dir, file)
                docs = self._extract_text(path)
                for d in docs:
                    d.metadata.setdefault('source', file)
                all_docs.extend(docs)
                self.metadata[file] = _hash_file(path)

            if not all_docs:
                return False, "No text extracted from documents."

            chunks = self.text_splitter.split_documents(all_docs)
            print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
            
            # Cache the chunks to avoid regenerating if embedding fails
            self._save_cached_chunks(chunks)
        
        # Create the index with OpenAI embeddings
        print("Creating index with OpenAI embeddings...")
        index = self._create_index(chunks)
        
        index_path = os.path.join(self.cache_dir, 'faiss_index')
        os.makedirs(index_path, exist_ok=True)
        index.save_local(index_path)

        self.index = index
        self._save_metadata()
        
        # Save the files hash
        with open(files_hash_file, 'w') as f:
            f.write(current_files_hash)
        
        # Save timestamp when completed successfully
        with open(timestamp_file, 'w') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(timestamp)
            
        return True, f"Indexed {len(files)} documents into {len(chunks)} chunks."

    @st.cache_data(show_spinner=False)
    def query(self, text: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on the indexed content."""
        if not self.index:
            self.process_documents()
        results = self.index.similarity_search(text, k=k)
        return [
            {'content': doc.page_content, 'source': doc.metadata.get('source'), 'page': doc.metadata.get('page')}
            for doc in results
        ]
