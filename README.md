# 📘 Simple RAG Pipeline with LangChain + Ollama
This implementation demonstrates a basic Retrieval-Augmented Generation (RAG) workflow using LangChain, a local Ollama LLM, and a Chroma vector database.

## 🚀 Overview

The application:

* Loads text documents from a local file
* Splits them into manageable chunks
* Converts text into embeddings using a Hugging Face model
* Stores embeddings in a Chroma vector database
* Retrieves relevant context based on user queries
* Uses a local LLM (via Ollama) to generate concise answers

## 🛠️ Installation

Install the required dependencies using pip:

```bash
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install chromadb
pip install sentence-transformers
```

## ⚙️ Requirements

* Python 3.11
* Ollama running locally (`http://localhost:11434`)
* A pulled model (e.g., `llama3.2:1B`)

## 📂 Project Structure

```
project/
│── Knowledge/
│   └── company_docs.txt
│── chroma_db/
│── main.py
```

## ▶️ How It Works

1. **Document Loading** – Reads text data from a file
2. **Text Splitting** – Breaks text into overlapping chunks
3. **Embedding** – Converts text into vector representations
4. **Storage** – Saves embeddings in Chroma DB
5. **Retrieval** – Finds relevant chunks for a query
6. **Generation** – Uses LLM to answer based on retrieved context

## 📌 Notes

* Ensure Ollama is running before starting the script
* Adjust `chunk_size` and `k` for better retrieval performance
* You can replace the document file with your own dataset

