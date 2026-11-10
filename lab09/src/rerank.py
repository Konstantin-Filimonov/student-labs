"""Второй этап поиска: переранжирование кандидатов.

feature        — простой учебный reranker на дополнительных признаках (без нейросети);
cross_encoder  — нейронный CrossEncoder (расширенное задание, нужен reranker_model в config.yaml).
"""
import re

import numpy as np

_STOP = {"что", "как", "при", "для", "или", "все", "его", "она", "они", "это", "какие", "какой",
         "какая", "когда", "кто", "где", "сколько", "после", "перед", "если", "быть", "есть",
         "нужно", "должен", "должна", "должны", "ли", "не", "на", "по", "из", "от", "до", "за"}


def _stems(text):
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return {w[:5] for w in words if w not in _STOP and (len(w) >= 3 or w.isdigit())}


def _exact_tokens(question):
    """Аббревиатуры и числа из вопроса (SAST, CVSS, 9.0), которые должны совпасть буквально."""
    return [t.lower() for t in re.findall(r"[A-Za-z]{2,}[A-Za-z0-9\-]*|\d+(?:[.,]\d+)?", question)]


def _minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


def feature_rerank(question, candidates, weights):
    q_stems = _stems(question)
    exact = _exact_tokens(question)
    dense = _minmax([c["score_dense"] for c in candidates])
    scores = []
    for d, c in zip(dense, candidates):
        text = c["text"].lower()
        coverage = len(q_stems & _stems(c["text"])) / len(q_stems) if q_stems else 0.0
        hit_exact = sum(t in text for t in exact) / len(exact) if exact else 0.0
        scores.append(weights["dense"] * d + weights["coverage"] * coverage + weights["exact"] * hit_exact)
    order = np.argsort(-np.asarray(scores), kind="stable")
    out = []
    for new_rank, i in enumerate(order, 1):
        item = dict(candidates[i])
        item["rank"], item["score"] = new_rank, float(scores[i])
        out.append(item)
    return out


def cross_encoder_rerank(question, candidates, model):
    scores = np.asarray(model.predict([(question, c["text"]) for c in candidates]))
    order = np.argsort(-scores, kind="stable")
    out = []
    for new_rank, i in enumerate(order, 1):
        item = dict(candidates[i])
        item["rank"], item["score"] = new_rank, float(scores[i])
        out.append(item)
    return out


def build_reranker(kind, cfg):
    """Возвращает функцию fn(question, candidates) -> список, либо None для rerank=none."""
    if kind == "none":
        return None
    if kind == "feature":
        weights = cfg["rerank_weights"]
        return lambda q, cands: feature_rerank(q, cands, weights)
    if kind == "cross_encoder":
        if not cfg.get("reranker_model"):
            raise RuntimeError("reranker_model не задан в config.yaml")
        from sentence_transformers import CrossEncoder  # ленивый импорт
        model = CrossEncoder(cfg["reranker_model"])
        return lambda q, cands: cross_encoder_rerank(q, cands, model)
    raise ValueError(f"Неизвестный rerank: {kind}")
