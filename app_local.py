"""RAG assistant using a FREE local model (no API key needed)."""
import chromadb, torch, streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return embedder, col, tok, llm

st.title("Knowledge-Aware RAG Assistant")
with st.spinner("Loading models (first time downloads about 3 GB)..."):
    embedder, col, tok, llm = load()

q = st.text_input("Ask a question")

if q:
    res = col.query(query_embeddings=embedder.encode([q]).tolist(), n_results=4)
    context = "\n\n".join(res["documents"][0])
    sources = {m["source"] for m in res["metadatas"][0]}

    messages = [
        {"role": "system", "content": "Answer ONLY using the context provided. "
                                      "If the answer is not in the context, say: I don't know based on the documents."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    with st.spinner("Thinking..."):
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=200, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    st.write(answer)
    st.caption("Sources: " + ", ".join(sources))
