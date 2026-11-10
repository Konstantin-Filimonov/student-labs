"""Тесты без нейросетей и LLM: py -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_encoder import HashEncoder  # noqa: E402
from evaluate import evaluate_experiment, question_metrics  # noqa: E402
from ingest import (build_index, chunk_text, load_config, load_corpus, load_questions,  # noqa: E402
                    make_chunks, validate_data)
from retrieve import retrieve  # noqa: E402

CFG = load_config()
DOCS, QUESTIONS = load_corpus(), load_questions()


class DataTests(unittest.TestCase):
    def test_data_valid(self):
        validate_data(DOCS, QUESTIONS)
        self.assertGreaterEqual(len(QUESTIONS), 20)

    def test_experiment_ids_unique(self):
        ids = [e["id"] for e in CFG["experiments"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(CFG["baseline"], ids)


class ChunkTests(unittest.TestCase):
    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 100, 100)

    def test_size_and_overlap(self):
        text = " ".join(f"слово{i}" for i in range(400))
        chunks = chunk_text(text, 300, 100)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        for a, b in zip(chunks, chunks[1:]):
            self.assertTrue(set(a.split()) & set(b.split()), "соседние фрагменты должны перекрываться")
        self.assertEqual(chunk_text(text, 300, 0)[0].split()[-1] in text, True)

    def test_more_chunks_for_smaller_size(self):
        n = [len(make_chunks(DOCS, s, 100)) for s in (300, 600, 1200)]
        self.assertGreater(n[0], n[1])
        self.assertGreater(n[1], n[2])

    def test_unique_chunk_ids(self):
        ids = [c["chunk_id"] for c in make_chunks(DOCS, 300, 100)]
        self.assertEqual(len(ids), len(set(ids)))


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        res = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
        m = question_metrics(res, ["b", "x"])
        self.assertEqual(m["hit"], 1)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["rr"], 0.5)  # дубли doc_id схлопываются: b на втором месте
        self.assertEqual(question_metrics(res, ["z"])["rr"], 0.0)


class RetrieveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HashEncoder()
        cls.index = build_index(DOCS, cls.model, 600, 100, CFG["lexical"])

    def exp(self, **kw):
        base = {"id": "t", "chunk_size": 600, "overlap": 100, "search_mode": "dense", "rerank": "none",
                "candidate_n": 12, "top_k": 4}
        base.update(kw)
        return base

    def test_acl_public_only(self):
        for mode in ("dense", "lexical", "hybrid"):
            r = retrieve("уязвимости CVSS SAST SOC блок-лист", self.model, self.index,
                         self.exp(search_mode=mode, top_k=8), user_group="public")
            self.assertTrue(all(x["access_group"] == "public" for x in r["results"]), mode)

    def test_acl_role_sees_own_group(self):
        r = retrieve("блок-лист подозрительный IP-адрес", self.model, self.index,
                     self.exp(top_k=3), user_group="soc")
        self.assertEqual(r["results"][0]["doc_id"], "r09")

    def test_rerank_keeps_candidate_set(self):
        from rerank import build_reranker
        e = self.exp(rerank="feature", search_mode="hybrid")
        r = retrieve("Что обязательно проверяется перед выпуском релиза SAST?", self.model, self.index, e,
                     CFG["rrf_c"], None, build_reranker("feature", CFG))
        self.assertEqual(len(r["candidates"]), 12)
        self.assertEqual(len(r["results"]), 4)
        self.assertTrue({x["chunk_id"] for x in r["results"]} <= {x["chunk_id"] for x in r["candidates"]})

    def test_pipeline_smoke(self):
        cache = {}
        for exp in CFG["experiments"]:
            if exp["rerank"] == "cross_encoder":
                continue
            row, runs = evaluate_experiment(exp, DOCS, QUESTIONS, self.model, CFG, cache)
            for key in ("hit_at_k", "recall_at_k", "mrr"):
                self.assertTrue(0.0 <= row[key] <= 1.0, (exp["id"], key))
            self.assertEqual(len(runs), len(QUESTIONS))


if __name__ == "__main__":
    unittest.main()
