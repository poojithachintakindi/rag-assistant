import time, requests, chromadb
from sentence_transformers import SentenceTransformer

TOPICS = ["machine learning", "deep learning", "neural network", "natural language processing",
          "computer vision", "reinforcement learning", "data science", "cloud computing",
          "docker software", "kubernetes", "cybersecurity", "smart grid"]
TARGET_DOCS = 500
CHUNK_SIZE, OVERLAP = 800, 100

HEADERS = {"User-Agent": "RAGLearningProject/1.0 (student project; contact: your_email@example.com)"}
API = "https://en.wikipedia.org/w/api.php"

def api(params):
    r = requests.get(API, params={**params, "format": "json"}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def chunk(text):
    step = CHUNK_SIZE - OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]

titles = []
for t in TOPICS:
    for offset in (0, 50):
        data = api({"action": "query", "list": "search", "srsearch": t,
                    "srlimit": 50, "sroffset": offset})
        for item in data["query"]["search"]:
            if item["title"] not in titles:
                titles.append(item["title"])
print(f"Found {len(titles)} candidate pages")

docs = []
for title in titles:
    try:
        data = api({"action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "redirects": 1})
        page = next(iter(data["query"]["pages"].values()))
        text = page.get("extract", "")
        if len(text) > 1500:
            docs.append((page["title"], text))
    except Exception as e:
        print("skip", title, e)
    time.sleep(0.1)
    if len(docs) % 50 == 0 and docs:
        print(f"  {len(docs)} downloaded...")
    if len(docs) >= TARGET_DOCS:
        break
print(f"Downloaded {len(docs)} documents")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("docs")

ids, texts, metas = [], [], []
for title, content in docs:
    for n, c in enumerate(chunk(content)):
        ids.append(f"{title}-{n}"); texts.append(c); metas.append({"source": title})

BATCH = 256
for i in range(0, len(texts), BATCH):
    col.add(ids=ids[i:i+BATCH], documents=texts[i:i+BATCH],
            embeddings=model.encode(texts[i:i+BATCH]).tolist(), metadatas=metas[i:i+BATCH])
print(f"Indexed {len(docs)} documents as {len(texts)} chunks")