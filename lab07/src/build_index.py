import json
import os
import yaml
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Корень lab07 — на уровень выше src/
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
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def build_index(config, docs):
    model = SentenceTransformer(config["model_name"])
    all_texts = []
    metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"], config["chunk_size"], config["chunk_overlap"])
        for i, ch in enumerate(chunks):
            all_texts.append(ch)
            metadata.append({
                "doc_id": doc["id"],
                "chunk_id": i,
                "source": doc.get("source", ""),
                "category": doc.get("category", "")
            })
    embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(RESULTS_DIR / "embeddings.npy", embeddings)
    with open(RESULTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    print(f"Индекс построен: {len(all_texts)} чанков, размерность {embeddings.shape[1]}")
    return model, embeddings, all_texts, metadata

if __name__ == "__main__":
    cfg = load_config()
    corpus = load_corpus()
    model, emb, texts, meta = build_index(cfg, corpus)