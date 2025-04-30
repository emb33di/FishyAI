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
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity


class SpacyEmbeddings:
    """
    Local embedding model using spaCy's word vectors
    """
    def __init__(self, model_name="en_core_web_md"):
        """Initialize with a specific spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Download the model if it's not available
            import subprocess
            print(f"Downloading spaCy model {model_name}...")
            subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=True)
            self.nlp = spacy.load(model_name)
            
        # Check if the model has word vectors
        if not self.nlp.has_pipe("tok2vec"):
            raise ValueError(f"The spaCy model '{model_name}' does not have word vectors")
            
        self.vectors = None
        self.texts = None
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents with improved handling"""
        # Process texts in batches for better performance
        docs = list(self.nlp.pipe(texts, batch_size=20, disable=["ner", "parser"]))
        
        # Calculate document embeddings (mean of word vectors)
        embeddings = []
        for doc in docs:
            # Skip empty docs
            if len(doc) == 0:
                embeddings.append(np.zeros(self.nlp.vocab.vectors.shape[1]))
                continue
                
            # Use mean of word vectors for document embedding
            doc_vector = doc.vector
            
            # Convert to list if it's a numpy array
            if isinstance(doc_vector, np.ndarray):
                embeddings.append(doc_vector)
            else:
                embeddings.append(doc_vector)
        
        # Store for future similarity searches
        self.texts = texts
        # Make sure vectors is a numpy array
        self.vectors = np.array(embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query text"""
        doc = self.nlp(text, disable=["ner", "parser"])
        return doc.vector.tolist()
    
    def similarity_search(self, query_embedding, k=5):
        """Perform similarity search between the query and the stored document vectors"""
        if self.vectors is None:
            raise ValueError("No documents have been embedded yet")
            
        # Convert query embedding to numpy array
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.vectors)[0]
        
        # Get indices of top k most similar documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_indices, similarities[top_indices]


class FAISSEquivalent:
    """
    A simplified FAISS-like interface using spaCy embeddings for compatibility
    """
    def __init__(self, embeddings: SpacyEmbeddings, documents: List[Document]):
        self.embeddings = embeddings
        self.documents = documents
        self.doc_texts = [doc.page_content for doc in documents]
        
        # Embed all documents
        self.embeddings.embed_documents(self.doc_texts)
    
    @classmethod
    def from_documents(cls, documents: List[Document], embeddings: SpacyEmbeddings):
        """Create a FAISSEquivalent instance from documents and embeddings"""
        return cls(embeddings, documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Find the k most similar documents to the query"""
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Get the indices and scores of the most similar documents
        indices, _ = self.embeddings.similarity_search(query_embedding, k=k)
        
        # Return the corresponding documents
        return [self.documents[idx] for idx in indices]
    
    def save_local(self, folder_path: str):
        """Save the index and documents to a local folder with improved serialization"""
        os.makedirs(folder_path, exist_ok=True)
        
        # Save documents
        with open(os.path.join(folder_path, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Convert numpy arrays to lists for more reliable serialization
        texts = self.embeddings.texts
        
        # Handle vectors - ensure we're saving in a format that can be reliably loaded
        vectors = None
        if self.embeddings.vectors is not None:
            if isinstance(self.embeddings.vectors, np.ndarray):
                vectors = self.embeddings.vectors.tolist()
            else:
                # If it's already something else, save as is
                vectors = self.embeddings.vectors
        
        # Save embeddings data with safer serialization
        with open(os.path.join(folder_path, 'embeddings.pkl'), 'wb') as f:
            pickle.dump({
                'texts': texts,
                'vectors': vectors
            }, f)
    
    @classmethod
    def load_local(cls, folder_path: str, embedding_instance: SpacyEmbeddings = None):
        """Load the index and documents from a local folder with improved deserialization"""
        # Load documents
        with open(os.path.join(folder_path, 'documents.pkl'), 'rb') as f:
            documents = pickle.load(f)
        
        # Load embeddings
        with open(os.path.join(folder_path, 'embeddings.pkl'), 'rb') as f:
            emb_data = pickle.load(f)
        
        # Create or update embeddings instance
        embeddings = embedding_instance or SpacyEmbeddings()
        embeddings.texts = emb_data['texts']
        
        # Convert vectors back to numpy array if they were saved as lists
        if emb_data['vectors'] is not None:
            embeddings.vectors = np.array(emb_data['vectors'])
        else:
            embeddings.vectors = None
        
        # Create and return the FAISSEquivalent instance
        return cls(embeddings, documents)


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
    and indexes content using spaCy embeddings with caching.
    """
    def __init__(
        self,
        pdf_dir: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        cache_subdir: str = ".cache",
        spacy_model: str = "en_core_web_md"  # Medium-sized English model with word vectors
    ):
        self.pdf_dir = pdf_dir
        self.cache_dir = os.path.join(pdf_dir, cache_subdir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use spaCy embeddings instead of TF-IDF
        self.spacy_model = spacy_model
        self.embeddings = self._get_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.index = self._load_index()
        self.meta_path = os.path.join(self.cache_dir, 'metadata.pkl')
        self.metadata = self._load_metadata()

    @st.cache_resource(show_spinner=False, ttl=3600)  # Add time-to-live to avoid stale caches
    def _get_embeddings(_self):
        """Get spaCy embeddings model"""
        try:
            import spacy
            return SpacyEmbeddings(model_name=_self.spacy_model)
        except Exception as e:
            print(f"Error loading embeddings model: {e}")
            # Fallback to a simpler model if available
            try:
                return SpacyEmbeddings(model_name="en_core_web_sm")
            except Exception as err:
                print(f"Failed to load fallback model: {err}")
                raise RuntimeError("Could not load any spaCy model with word vectors")

    @st.cache_resource(ttl=3600)  # Add time-to-live to avoid stale caches
    def _load_index(_self) -> FAISSEquivalent:
        """Load the index with better error handling"""
        try:
            index_path = os.path.join(_self.cache_dir, 'faiss_index')
            if os.path.exists(index_path):
                try:
                    return FAISSEquivalent.load_local(index_path, _self.embeddings)
                except Exception as e:
                    print(f"Error loading index: {e}")
                    # If embedding dimension changed, we need to reindex
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

    def _needs_reindex(self, files: List[str]) -> bool:
        if not self.index:
            return True
        for pdf in files:
            path = os.path.join(self.pdf_dir, pdf)
            if self.metadata.get(pdf) != _hash_file(path):
                return True
        return False

    def _extract_text(self, path: str) -> List[Document]:
        # Existing implementation - no changes needed
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
        """Load, split, and index all PDFs, using cache when valid."""
        if not os.path.isdir(self.pdf_dir):
            return False, f"Directory not found: {self.pdf_dir}"

        pdfs = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        if not pdfs:
            return False, "No PDFs found."

        if not self._needs_reindex(pdfs):
            return True, f"Using cached index for {len(pdfs)} PDFs."

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
        
        # Print info about chunking
        print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
        
        # Create the index with spaCy embeddings
        print("Creating index with spaCy embeddings...")
        index = FAISSEquivalent.from_documents(chunks, self.embeddings)
        
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
