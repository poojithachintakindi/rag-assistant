"""Step 2: ask questions. Retrieves top chunks, sends them to the LLM."""
import os, chromadb, streamlit as st
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

@st.cache_resource
def load():
    return (SentenceTransformer("all-MiniLM-L6-v2"),
            chromadb.PersistentClient(path="chroma_db").get_collection("docs"),
            Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))

embedder, col, llm = load()
st.title("Knowledge-Aware RAG Assistant")
q = st.text_input("Ask a question")

if q:
    res = col.query(query_embeddings=embedder.encode([q]).tolist(), n_results=4)
    context = "\n\n".join(res["documents"][0])
    sources = {m["source"] for m in res["metadatas"][0]}
    prompt = (f"Answer ONLY from the context below. If the answer is not there, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {q}")
    out = llm.messages.create(model="claude-sonnet-5-5", max_tokens=500,
                              messages=[{"role": "user", "content": prompt}])
    st.write(out.content[0].text)
    st.caption("Sources: " + ", ".join(sources))
