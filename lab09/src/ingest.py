"""Загрузка конфигурации и данных, разбиение на фрагменты, построение индекса.

Размер фрагмента и перекрытие задаются в СИМВОЛАХ. Для каждого фрагмента сохраняется
устойчивый doc_id документа: метрики считаются по doc_id, а не по chunk_id.

Запуск (без моделей): py src/ingest.py  -> results/chunk_stats.csv
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
CORPUS_PATH = BASE_DIR / "data" / "corpus.jsonl"
QUESTIONS_PATH = BASE_DIR / "data" / "questions.jsonl"

RESULTS_DIR.mkdir(exist_ok=True)


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path=CORPUS_PATH):
    return read_jsonl(path)


def load_questions(path=QUESTIONS_PATH):
    return read_jsonl(path)


def corpus_fingerprint(paths=(CORPUS_PATH, QUESTIONS_PATH)):
    """Короткий хеш входных данных для журнала экспериментов."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


def validate_data(docs, questions):
    """Проверка корпуса и разметки (требования методички, п. 3.2)."""
    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id в корпусе должны быть уникальными")
    known = set(ids)
    for q in questions:
        if q["answerable"]:
            if not q["gold_doc_ids"]:
                raise ValueError(f"{q['id']}: у ответимого вопроса нет gold_doc_ids")
            missing = set(q["gold_doc_ids"]) - known
            if missing:
                raise ValueError(f"{q['id']}: неизвестные doc_id {sorted(missing)}")
        elif q["gold_doc_ids"]:
            raise ValueError(f"{q['id']}: у неответимого вопроса не должно быть gold_doc_ids")
    n_ans = sum(q["answerable"] for q in questions)
    if n_ans < 15 or len(questions) - n_ans < 5:
        raise ValueError("Нужно не менее 15 ответимых и 5 неответимых вопросов")


def chunk_text(text, size=600, overlap=100):
    """Разбиение по символам с переносом границы на ближайший пробел (листинг 1 методички)."""
    if not 0 <= overlap < size:
        raise ValueError("Требуется 0 <= overlap < size")
    text = " ".join(text.split())
    pos, result = 0, []
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", pos + size // 2, end)
            if cut > pos:
                end = cut
        fragment = text[pos:end].strip()
        if fragment:
            result.append(fragment)
        if end >= len(text):
            break
        pos = max(pos + 1, end - overlap)
    return result


def make_chunks(docs, size=600, overlap=100):
    chunks = []
    for doc in docs:
        for i, fragment in enumerate(chunk_text(doc["text"], size, overlap)):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "source": doc.get("title", doc["doc_id"]),
                "category": doc.get("category", ""),
                "access_group": doc.get("access_group", "public"),
                "text": fragment,
            })
    return chunks


@dataclass
class Index:
    chunks: list
    dense: np.ndarray
    tfidf: TfidfVectorizer
    lexical: object
    size: int
    overlap: int
    extra: dict = field(default_factory=dict)


def get_encoder(model_name):
    from sentence_transformers import SentenceTransformer  # ленивый импорт: тяжёлая зависимость
    return SentenceTransformer(model_name)


def build_index(docs, model, size, overlap, lexical_cfg=None):
    lexical_cfg = lexical_cfg or {"analyzer": "char", "ngram_range": [3, 5]}
    chunks = make_chunks(docs, size, overlap)
    texts = [c["text"] for c in chunks]
    dense = np.asarray(model.encode(texts, normalize_embeddings=True))
    tfidf = TfidfVectorizer(analyzer=lexical_cfg["analyzer"],
                            ngram_range=tuple(lexical_cfg["ngram_range"]))
    lexical = tfidf.fit_transform(texts)
    return Index(chunks, dense, tfidf, lexical, size, overlap)


def chunk_stats(docs, pairs):
    rows = []
    for size, overlap in pairs:
        chunks = make_chunks(docs, size, overlap)
        lens = [len(c["text"]) for c in chunks]
        rows.append({"chunk_size": size, "overlap": overlap, "n_chunks": len(chunks),
                     "avg_chunk_chars": round(sum(lens) / len(lens), 1),
                     "max_chunk_chars": max(lens),
                     "total_chars_indexed": sum(lens)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()
    documents, questions = load_corpus(), load_questions()
    validate_data(documents, questions)
    pairs = sorted({(e["chunk_size"], e["overlap"]) for e in cfg["experiments"]})
    df = chunk_stats(documents, pairs)
    df.to_csv(RESULTS_DIR / "chunk_stats.csv", index=False)
    print(f"Документов: {len(documents)}, вопросов: {len(questions)}")
    print(df.to_string(index=False))
