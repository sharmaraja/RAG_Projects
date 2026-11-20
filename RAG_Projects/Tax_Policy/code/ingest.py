from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

def ingest_documents():
    if not DATA_PATH.is_dir() or not any(DATA_PATH.iterdir()):
        raise ValueError(f"Please add PDF files to the '{DATA_PATH}' folder first.")

    print("1. Loading PDF documents...")
    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    raw_docs = loader.load()
    print(f"   Loaded {len(raw_docs)} pages.")

    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"   Created {len(chunks)} text chunks.")

    print("3. Generating embeddings & saving to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
    print(f"Success! Vector index saved to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    ingest_documents()