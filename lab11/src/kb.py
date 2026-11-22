"""Реализации search_knowledge: интерфейс search(query, user_group) -> [{id, doc_id, source, text, score}].

KeywordKB — детерминированный учебный поиск (пересечение основ слов; НЕ семантический).
Lab09KB   — поиск из ЛР9 (dense / lexical / hybrid + RRF, ACL по access_group), интерфейс тот же (этап 6 методички).
Содержимое найденных документов — НЕДОВЕРЕННЫЕ ДАННЫЕ: агент передаёт их модели только в role=tool.
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
        self.docs = docs if docs is not None else read_jsonl(BASE_DIR / "data" / "kb.jsonl")
        self.top = top

    def search(self, query, user_group=None):
        q = _stems(query)
        scored = []
        for d in self.docs:
            if not _allowed(d.get("access_group", "public"), user_group):
                continue
            n = len(q & _stems(d["text"] + " " + d.get("title", "")))
            if n > 0:
                scored.append((n, d))
        scored.sort(key=lambda x: -x[0])  # стабильная сортировка: при равенстве порядок документов
        return [{"id": d["doc_id"], "doc_id": d["doc_id"], "source": d.get("title", d["doc_id"]),
                 "text": d["text"][:MAX_TEXT], "score": n} for n, d in scored[: self.top]]


class Lab09KB:
    name = "lab09"

    def __init__(self, lab09_src, experiment_id="hybrid_600_100_k4", extra_docs=None, encoder=None):
        src = Path(lab09_src).resolve()
        if not (src / "ingest.py").exists():
            raise FileNotFoundError(f"Не найден код ЛР9: {src}")
        if str(src) not in sys.path:
            sys.path.append(str(src))  # в конец: модули ЛР11 имеют приоритет
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

    def search(self, query, user_group=None):
        r = self._retrieve.retrieve(query, self.model, self.index, self.exp, self.cfg["rrf_c"],
                                    user_group, self.reranker)
        return [{"id": x["chunk_id"], "doc_id": x["doc_id"], "source": x["source"],
                 "text": x["text"][:MAX_TEXT], "score": round(x["score"], 4)} for x in r["results"]]


def make_kb(backend="keyword", extra_docs=None, lab09_src=None, experiment_id="hybrid_600_100_k4", encoder=None):
    """Создаёт KB. При недоступности ЛР9 честно сообщает в stderr и переключается на KeywordKB."""
    if backend == "lab09":
        try:
            return Lab09KB(lab09_src or (BASE_DIR / ".." / "lab09" / "src"), experiment_id, extra_docs, encoder)
        except (ImportError, FileNotFoundError, KeyError) as exc:
            print(f"[kb] ЛР9 недоступна ({exc}); используется keyword-заменитель", file=sys.stderr)
    docs = read_jsonl(BASE_DIR / "data" / "kb.jsonl") + list(extra_docs or [])
    return KeywordKB(docs)


def load_poisoned_docs():
    return read_jsonl(BASE_DIR / "data" / "kb_poisoned.jsonl")
