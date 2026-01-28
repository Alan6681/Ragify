from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

class AIService:
    def __init__(self, files):
        self.files = files
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def load_pdf(self):
        documents = []

        for file in self.files:
            file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

            os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        return splitter.split_documents(documents)

    def create_vectorstore(self):
        if self.vectorstore is None:
            docs = self.load_pdf()
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return self.vectorstore

    def create_retriever(self):
        return self.create_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    def load_llm(self):
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )
    
    
