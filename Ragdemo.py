from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Load Local LLM (Ollama)
# -----------------------------
llm = ChatOllama(
    model="llama3.2:1B",
    base_url="http://localhost:11434",
    temperature=0.1,
)
print("LLM loaded")

# -----------------------------
# Load Documents
# -----------------------------
DATA_PATH = "Knowledge/company_docs.txt"

loader = TextLoader(DATA_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# -----------------------------
# Split Documents into Chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # larger chunks for better context
    chunk_overlap=100     # overlap for continuity
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# -----------------------------
# Create Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded")

# -----------------------------
# Create Vector Store
# -----------------------------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vector_store.persist()
print("Vector store created and persisted")

# -----------------------------
# RAG Function
# -----------------------------
def rag(query):
    # Retrieve top 3 relevant chunks
    relevant_chunks = vector_store.similarity_search(query, k=3)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    # Smarter prompt: summarize relevant points concisely
    prompt = f"""
You are an AI assistant. Based on the context below, answer the question concisely.
Only use information from the context. 
Context:
{context}

Question:
{query}

Answer:
"""

    # Generate response
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content  # prints only once

# -----------------------------
# Interactive Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")

        if user_query.lower() == "exit":
            break

        answer = rag(user_query)
        print(f"\nFinal Answer: {answer}")  # prints only once