import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

load_dotenv()

PDF_PATH = "medical.pdf"  # 🔹 your PDF file
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found. Place your medical PDF in the project folder.")

    print("[Ingest] Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"[Ingest] Loaded {len(docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print(f"[Ingest] Total chunks: {len(chunks)}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[Ingest] Creating Chroma vector database...")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    print(f"[Ingest] ✅ Ingestion complete. Data stored in: {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_pdf()
