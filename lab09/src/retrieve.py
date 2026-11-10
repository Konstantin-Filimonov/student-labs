"""Поиск: плотный, лексический (TF-IDF) и гибридный (RRF); фильтр прав доступа (ACL)."""
import time

import numpy as np


def _allowed_ids(index, user_group):
    """ACL-фильтр применяется ДО ранжирования и передачи фрагментов в LLM."""
    return np.array([i for i, c in enumerate(index.chunks)
                     if user_group is None or c["access_group"] in (user_group, "public")], dtype=int)


def _positions(order, n):
    ranks = np.full(n, n + 1, dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def first_stage(question, model, index, mode, rrf_c=60, user_group=None):
    """Возвращает упорядоченные индексы фрагментов и три вектора оценок."""
    q = np.asarray(model.encode([question], normalize_embeddings=True)[0])
    dense = index.dense @ q
    lex = (index.lexical @ index.tfidf.transform([question]).T).toarray().ravel()
    ids = _allowed_ids(index, user_group)
    n = len(index.chunks)
    dense_order = ids[np.argsort(-dense[ids], kind="stable")]
    lex_order = ids[np.argsort(-lex[ids], kind="stable")]
    if mode == "dense":
        return dense_order, dense, lex, dense
    if mode == "lexical":
        return lex_order, dense, lex, lex
    if mode == "hybrid":
        rd, rl = _positions(dense_order, n), _positions(lex_order, n)
        fused = 1.0 / (rrf_c + rd) + 1.0 / (rrf_c + rl)  # ранговая, а не вероятностная оценка
        return ids[np.argsort(-fused[ids], kind="stable")], dense, lex, fused
    raise ValueError(f"Неизвестный search_mode: {mode}")


def _pack(index, i, rank, dense, lex, score):
    c = index.chunks[i]
    return {"rank": rank, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "source": c["source"],
            "category": c["category"], "access_group": c["access_group"], "text": c["text"],
            "score": float(score[i]), "score_dense": float(dense[i]), "score_lex": float(lex[i])}


def retrieve(question, model, index, exp, rrf_c=60, user_group=None, rerank_fn=None):
    """Двухэтапный поиск. Без переранжирования кандидаты = итоговый top-k."""
    use_rerank = exp.get("rerank", "none") != "none"
    if use_rerank and exp["candidate_n"] < exp["top_k"]:
        raise ValueError("candidate_n должен быть >= top_k")
    t0 = time.perf_counter()
    order, dense, lex, score = first_stage(question, model, index, exp["search_mode"], rrf_c, user_group)
    n_first = exp["candidate_n"] if use_rerank else exp["top_k"]
    candidates = [_pack(index, i, r, dense, lex, score) for r, i in enumerate(order[:n_first], 1)]
    t_search = time.perf_counter() - t0

    t_rerank, final = 0.0, candidates
    if use_rerank:
        t1 = time.perf_counter()
        final = rerank_fn(question, candidates)[: exp["top_k"]]
        t_rerank = time.perf_counter() - t1
    return {"candidates": candidates, "results": final,
            "t_search_s": t_search, "t_rerank_s": t_rerank}
