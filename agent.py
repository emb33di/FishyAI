import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

class PropertyLawAgent:
    def __init__(self, pdf_directory: str):
        load_dotenv()
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
        self.chat_history = []
        
    def load_pdfs(self):
        """Load and process all PDFs in the specified directory."""
        documents = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
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
        
        # Initialize QA chain
        llm = ChatOpenAI(temperature=0)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> str:
        """Ask a question about property law and get an answer based on the provided context."""
        if not self.qa_chain:
            return "Please load PDFs first using load_pdfs()"
        
        result = self.qa_chain({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, result["answer"]))
        return result["answer"]
