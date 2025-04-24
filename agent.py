import os
from typing import List, Tuple, Dict
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st

class PropertyLawAgent:
    def __init__(self, pdf_directory: str):
        # Load environment variables
        load_dotenv()
        
        # Try to get API key from Streamlit secrets first, then from .env
        api_key = st.secrets.get("secrets", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or .env file")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key
        )
        
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None
        self.qa_chain = None
        self.chat_history = []
        self.loaded_pdfs = []
        
    def load_pdfs(self) -> Tuple[bool, str]:
        """Load and process all PDFs in the specified directory."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                return False, f"Created empty '{self.pdf_directory}' directory. Please add your PDF files there."
            
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                return False, f"No PDF files found in '{self.pdf_directory}' directory. Please add your PDF files."
            
            documents = []
            self.loaded_pdfs = []
            
            print("\nLoading PDFs:")
            for filename in pdf_files:
                print(f"- Processing {filename}...")
                file_path = os.path.join(self.pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                self.loaded_pdfs.append(filename)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            # Initialize QA chain with GPT-4
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
                max_tokens=2000
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            return True, f"Successfully loaded {len(self.loaded_pdfs)} PDFs: {', '.join(self.loaded_pdfs)}"
        except Exception as e:
            return False, f"Error loading PDFs: {str(e)}"
    
    def ask_question(self, question: str) -> Dict[str, str]:
        """Ask a question about property law and get an answer based on the provided context."""
        if not self.qa_chain:
            return {
                "answer": "Please load PDFs first using load_pdfs()",
                "sources": []
            }
        
        try:
            result = self.qa_chain({"question": question, "chat_history": self.chat_history})
            
            # Extract source documents
            sources = []
            for doc in result.get("source_documents", []):
                source_file = doc.metadata.get("source", "").split("/")[-1]
                page_num = doc.metadata.get("page", 1)
                if source_file and source_file not in sources:
                    sources.append(f"{source_file} (page {page_num})")
            
            self.chat_history.append((question, result["answer"]))
            
            return {
                "answer": result["answer"],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    
    def get_loaded_pdfs(self) -> List[str]:
        """Return list of currently loaded PDFs."""
        return self.loaded_pdfs

    def get_embeddings(self, texts):
        """Get embeddings for a list of texts using direct OpenAI API call"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
