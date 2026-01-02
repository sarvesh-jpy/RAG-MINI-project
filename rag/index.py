import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Paths
# Make sure your PDF is actually named "nodejs.pdf"
pdf_path = Path(__file__).parent / "nodejs.pdf" 

def main():
    if not pdf_path.exists():
        print(f"❌ Error: File not found at {pdf_path}")
        return

    print("📄 Loading PDF...")
    loader = PyPDFLoader(file_path=str(pdf_path))
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")

    print("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents=docs)
    print(f"✅ Created {len(chunks)} text chunks.")

    print("🧠 Creating Embeddings (this runs locally)...")
    # Using the free local model to avoid API key errors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Saving to Qdrant...")
    try:
        # Connects to Docker running on port 6333
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name="learning_docs",
        )
        print("🎉 Success! Vector Store created.")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("👉 Is Docker running? Run: 'docker run -p 6333:6333 qdrant/qdrant'")

if __name__ == "__main__":
    main()