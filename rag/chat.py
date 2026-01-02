import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
# These are the correct modern imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load Keys
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    exit()

# 2. Setup The Brains
print("🔌 Connecting to Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
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

# 3. Build the Chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If you don't know the answer, say "I don't know".

<context>
{context}
</context>

Question: {input}
""")

# Create the chain that combines documents + LLM
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain (finds docs -> passes to document_chain)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. Chat Loop
print("\n🤖 DocuMind Ready! (Type 'exit' to quit)")
print("=" * 40)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    if user_input.strip() == "":
        continue

    print("Thinking...")
    try:
        response = retrieval_chain.invoke({"input": user_input})
        print(f"DocuMind: {response['answer']}")
    except Exception as e:
        print(f"❌ Error generating response: {e}")