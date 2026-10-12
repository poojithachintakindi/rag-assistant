# RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions
from 500 Wikipedia articles on AI, cloud, and security topics.

## How it works
1. Download 500 documents
2. Split them into 12,813 chunks (800 characters, 100 overlap)
3. Embed each chunk with all-MiniLM-L6-v2
4. Store the vectors in ChromaDB
5. For each question, retrieve the top 4 chunks and give them to an LLM,
   which answers only from that context

## Run it
pip install -r requirements.txt
python ingest.py
streamlit run app_local.py --server.fileWatcherType none

## Tech
Python, ChromaDB, sentence-transformers, Streamlit, Qwen2.5 (local) / Claude API
