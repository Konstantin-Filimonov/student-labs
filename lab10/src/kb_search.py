"""Реализации search_knowledge_base: интерфейс search(query, user_group) -> [{id, doc_id, source, text}].

KeywordKB — учебный заменитель из методички (пересечение слов, НЕ семантический поиск).
Lab09KB   — поиск из ЛР9 (dense / lexical / hybrid + RRF, ACL по access_group), без изменения интерфейса.
Формат документов единый (как corpus.jsonl в ЛР9): doc_id, title, category, access_group, text.
"""
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MAX_TEXT = 600


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _allowed(doc_group, user_group):
    return user_group is None or doc_group in (user_group, "public")


def _stems(text):
    """Слова и грубые основы (первые 5 букв): «инциденте» ~ «инцидент»."""
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) >= 3}


class KeywordKB:
    name = "keyword"

    def __init__(self, docs=None, top=2):
        self.docs = docs if docs is not None else read_jsonl(BASE_DIR / "data" / "kb_fallback.jsonl")
        self.top = top

    def search(self, query, user_group=None):
        q = _stems(query)
        scored = []
        for d in self.docs:
            if not _allowed(d.get("access_group", "public"), user_group):
                continue
            n = len(q & _stems(d["text"]))
            if n > 0:
                scored.append((n, d))
        scored.sort(key=lambda x: -x[0])  # sort стабилен: при равенстве сохраняется порядок документов
        return [{"id": d["doc_id"], "doc_id": d["doc_id"], "source": d.get("title", d["doc_id"]),
                 "text": d["text"][:MAX_TEXT], "score": n} for n, d in scored[: self.top]]


class Lab09KB:
    name = "lab09"

    def __init__(self, lab09_src, experiment_id, extra_docs=None, encoder=None):
        src = Path(lab09_src).resolve()
        if not (src / "ingest.py").exists():
            raise FileNotFoundError(f"Не найден код ЛР9: {src}")
        if str(src) not in sys.path:
            sys.path.append(str(src))  # в конец: модули ЛР10 имеют приоритет
        import ingest  # модули ЛР9
        import rerank
        import retrieve
        self._retrieve = retrieve
        root = src.parent
        self.cfg = ingest.load_config(root / "config" / "config.yaml")
        exps = {e["id"]: e for e in self.cfg["experiments"]}
        if experiment_id not in exps:
            raise KeyError(f"В ЛР9 нет конфигурации {experiment_id}")
        self.exp = exps[experiment_id]
        docs = ingest.load_corpus(root / "data" / "corpus.jsonl") + list(extra_docs or [])
        self.model = encoder if encoder is not None else ingest.get_encoder(self.cfg["model_name"])
        self.index = ingest.build_index(docs, self.model, self.exp["chunk_size"], self.exp["overlap"],
                                        self.cfg.get("lexical"))
        self.reranker = rerank.build_reranker(self.exp.get("rerank", "none"), self.cfg)
        self.experiment_id = experiment_id

    def search(self, query, user_group=None):
        r = self._retrieve.retrieve(query, self.model, self.index, self.exp, self.cfg["rrf_c"],
                                    user_group, self.reranker)
        return [{"id": x["chunk_id"], "doc_id": x["doc_id"], "source": x["source"],
                 "text": x["text"][:MAX_TEXT], "score": round(x["score"], 4)} for x in r["results"]]


def make_kb(cfg, extra_docs=None, backend=None, encoder=None):
    """Создаёт KB по config.yaml; при недоступности ЛР9 честно переключается на KeywordKB."""
    backend = backend or cfg["kb"]["backend"]
    fallback_docs = read_jsonl(BASE_DIR / "data" / "kb_fallback.jsonl") + list(extra_docs or [])
    if backend == "lab09":
        try:
            return Lab09KB(BASE_DIR / cfg["kb"]["lab09_src"], cfg["kb"]["lab09_experiment"], extra_docs, encoder)
        except (ImportError, FileNotFoundError, KeyError) as exc:
            print(f"[kb] ЛР9 недоступна ({exc}); используется keyword-заменитель", file=sys.stderr)
    return KeywordKB(fallback_docs)
