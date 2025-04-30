import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Any
from pdf2image import convert_from_path
import pytesseract
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import numpy as np


def _hash_file(path: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class PDFProcessor:
    """
    Extracts text from PDFs (with OCR fallback), splits into chunks,
    and indexes content using OpenAI embeddings with cost-saving caching.
    """
    def __init__(
        self,
        pdf_dir: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        cache_subdir: str = ".cache"
    ):
        self.pdf_dir = pdf_dir
        self.cache_dir = os.path.join(pdf_dir, cache_subdir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Get embeddings model - cached to prevent multiple initializations
        self.embeddings = self._get_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.index = self._load_index()
        self.meta_path = os.path.join(self.cache_dir, 'metadata.pkl')
        self.metadata = self._load_metadata()
        self.chunks_cache_path = os.path.join(self.cache_dir, 'chunks.pkl')

    @st.cache_resource(show_spinner=False, ttl=3600)
    def _get_embeddings(_self):
        """Get OpenAI embeddings model"""
        try:
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {e}")
            raise RuntimeError("Failed to initialize OpenAI embeddings")

    @st.cache_resource(ttl=3600)
    def _load_index(_self):
        """Load the index with error handling"""
        try:
            index_path = os.path.join(_self.cache_dir, 'faiss_index')
            if os.path.exists(index_path):
                try:
                    return FAISS.load_local(index_path, _self.embeddings)
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
        if not self.index:
            return True
        for pdf in files:
            path = os.path.join(self.pdf_dir, pdf)
            if self.metadata.get(pdf) != _hash_file(path):
                return True
        return False

    def _extract_text(self, path: str) -> List[Document]:
        """Extract text from PDFs with multiple fallback methods"""
        try:
            print(f"Attempting to extract text from: {path}")
            
            # Try unstructured loader first for PowerPoint-converted PDFs
            try:
                print("Attempting UnstructuredPDFLoader extraction...")
                # Try different modes with specific settings for PowerPoint PDFs
                for mode in ['elements', 'single', 'paged']:
                    try:
                        docs = UnstructuredPDFLoader(
                            path,
                            mode=mode,
                            strategy="fast",  # Use fast strategy for better handling of PowerPoint PDFs
                            include_metadata=True
                        ).load()
                        if any(d.page_content.strip() for d in docs):
                            print(f"Successfully extracted {len(docs)} pages using UnstructuredPDFLoader with mode {mode}")
                            # Clean up the extracted text
                            for doc in docs:
                                # Remove extra whitespace and normalize line breaks
                                doc.page_content = ' '.join(doc.page_content.split())
                            return docs
                    except Exception as e:
                        print(f"UnstructuredPDFLoader extraction failed with mode {mode}: {str(e)}")
                        continue
            except Exception as e:
                print(f"All UnstructuredPDFLoader attempts failed: {str(e)}")

            # Try native extraction as fallback
            try:
                print("Attempting PyPDFLoader extraction...")
                docs = PyPDFLoader(path).load()
                if any(d.page_content.strip() for d in docs):
                    print(f"Successfully extracted {len(docs)} pages using PyPDFLoader")
                    # Clean up the extracted text
                    for doc in docs:
                        doc.page_content = ' '.join(doc.page_content.split())
                    return docs
                print("PyPDFLoader extraction returned empty content")
            except Exception as e:
                print(f"PyPDFLoader extraction failed: {str(e)}")

            # OCR as last resort
            try:
                print("Attempting OCR extraction...")
                pages = convert_from_path(path, dpi=300)  # Higher DPI for better text recognition
                ocr_docs = []
                for i, img in enumerate(pages, start=1):
                    try:
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            # Clean up OCR text
                            text = ' '.join(text.split())
                            ocr_docs.append(Document(
                                page_content=text,
                                metadata={'source': os.path.basename(path), 'page': i}
                            ))
                            print(f"Successfully OCR'd page {i}")
                    except Exception as e:
                        print(f"Error processing page {i} of {path}: {str(e)}")
                        continue
                if ocr_docs:
                    print(f"Successfully OCR'd {len(ocr_docs)} pages")
                    return ocr_docs
                print("OCR extraction returned no content")
            except Exception as e:
                print(f"Error in OCR processing of {path}: {str(e)}")

            print(f"All extraction methods failed for {path}")
            return []

        except Exception as e:
            print(f"Error processing PDF {path}: {str(e)}")
            return []

    def process_pdfs(self) -> Tuple[bool, str]:
        """Load, split, and index all PDFs, using cache when valid to minimize API calls."""
        if not os.path.isdir(self.pdf_dir):
            return False, f"Directory not found: {self.pdf_dir}"

        pdfs = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        if not pdfs:
            return False, "No PDFs found."

        # If index exists and no PDFs have changed, use cached index
        if not self._needs_reindex(pdfs):
            return True, f"Using cached index for {len(pdfs)} PDFs."

        # Check if we have cached chunks that we can reuse
        cached_chunks = self._load_cached_chunks()
        if cached_chunks and all(pdf in self.metadata for pdf in pdfs):
            # Only use cached chunks if we have all PDFs accounted for
            chunks = cached_chunks
            print(f"Using {len(chunks)} cached chunks to avoid regenerating embeddings")
        else:
            # Process PDFs from scratch
            all_docs: List[Document] = []
            for pdf in pdfs:
                path = os.path.join(self.pdf_dir, pdf)
                docs = self._extract_text(path)
                for d in docs:
                    d.metadata.setdefault('source', pdf)
                all_docs.extend(docs)
                self.metadata[pdf] = _hash_file(path)

            if not all_docs:
                return False, "No text extracted from PDFs."

            chunks = self.text_splitter.split_documents(all_docs)
            print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
            
            # Cache the chunks to avoid regenerating if embedding fails
            self._save_cached_chunks(chunks)
        
        # Create the index with OpenAI embeddings
        print("Creating index with OpenAI embeddings...")
        index = FAISS.from_documents(chunks, self.embeddings)
        
        index_path = os.path.join(self.cache_dir, 'faiss_index')
        os.makedirs(index_path, exist_ok=True)
        index.save_local(index_path)

        self.index = index
        self._save_metadata()
        return True, f"Indexed {len(pdfs)} PDFs into {len(chunks)} chunks."

    def query(self, text: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on the indexed content."""
        if not self.index:
            self.process_pdfs()
        results = self.index.similarity_search(text, k=k)
        return [
            {'content': doc.page_content, 'source': doc.metadata.get('source'), 'page': doc.metadata.get('page')}
            for doc in results
        ]
