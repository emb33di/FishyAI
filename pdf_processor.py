# pdf_processor.py

import os
import pickle
import hashlib
from typing import List, Dict, Tuple
from pdf2image import convert_from_path
import pytesseract
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st


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
    and indexes content in a FAISS vector store with caching.
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

        self.embeddings = self._get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.index = self._load_index()
        self.meta_path = os.path.join(self.cache_dir, 'metadata.pkl')
        self.metadata = self._load_metadata()

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _get_embeddings() -> OpenAIEmbeddings:
        return OpenAIEmbeddings()

    @st.cache_resource
    def _load_index(_self=None) -> FAISS:
        index_path = os.path.join(self.cache_dir, 'faiss_index')
        if os.path.exists(index_path):
            try:
                return FAISS.load_local(index_path, self.embeddings)
            except Exception:
                os.remove(index_path)
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
        # Try native extraction
        docs = PyPDFLoader(path).load()
        if any(d.page_content.strip() for d in docs):
            return docs

        # Try unstructured loader
        docs = UnstructuredPDFLoader(path, mode='elements').load()
        if any(d.page_content.strip() for d in docs):
            return docs

        # OCR fallback
        pages = convert_from_path(path)
        ocr_docs = []
        for i, img in enumerate(pages, start=1):
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_docs.append(Document(
                    page_content=text,
                    metadata={'source': os.path.basename(path), 'page': i}
                ))
        return ocr_docs

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
