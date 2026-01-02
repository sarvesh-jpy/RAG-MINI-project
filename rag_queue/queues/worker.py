import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    exit()

# 2. Setup The Brains (Initialize Globals)
print("🔌 Connecting to Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    # Connection to Qdrant is still synchronous during setup (standard for this library)
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="learning_docs",
        url="http://localhost:6333",
    )
except Exception as e:
    print(f"❌ Error: Could not connect to Qdrant. {e}")
    exit()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=GROQ_API_KEY
)

# 3. Build the Async Chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you don't know the answer, say "I don't know".

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. Define Async Worker Function
async def process_query(user_input: str):
    """
    This function handles the asynchronous RAG processing.
    """
    try:
        # NOTICE: We use .ainvoke() instead of .invoke()
        response = await retrieval_chain.ainvoke({"input": user_input})
        return response['answer']
    except Exception as e:
        return f"❌ Error generating response: {e}"

# 5. Async Main Loop
async def main():
    print("\n🤖 DocuMind Ready! (Type 'exit' to quit)")
    print("=" * 40)

    while True:
        # Note: 'input()' is technically blocking, but fine for a CLI script.
        # In a real web server, you wouldn't use input() anyway.
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.strip() == "":
            continue

        print("Thinking...")
        
        # Await the response
        answer = await process_query(user_input)
        print(f"DocuMind: {answer}")

if __name__ == "__main__":
    # Start the async event loop
    asyncio.run(main())