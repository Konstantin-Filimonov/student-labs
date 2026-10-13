import json
import yaml
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
CORPUS_PATH = BASE_DIR / "data" / "corpus.jsonl"

RESULTS_DIR.mkdir(exist_ok=True)

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_corpus(path=CORPUS_PATH):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def build_index(config, docs):
    model = SentenceTransformer(config["model_name"])
    texts, meta = [], []
    for doc in docs:
        for i, ch in enumerate(chunk_text(doc["text"], config["chunk_size"], config["chunk_overlap"])):
            texts.append(ch)
            meta.append({
                "chunk_id": f"{doc['id']}_{i}",
                "doc_id": doc["id"],
                "source": doc.get("source", ""),
                "category": doc.get("category", ""),
                "access_group": doc.get("access_group", "public"),
            })
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(RESULTS_DIR / "embeddings.npy", emb)
    (RESULTS_DIR / "chunks.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Индекс: {len(texts)} чанков, dim={emb.shape[1]}")
    return model, emb, texts, meta

if __name__ == "__main__":
    cfg = load_config()
    build_index(cfg, load_corpus())