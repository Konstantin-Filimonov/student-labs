"""Шаги 4, 6 и 7 методички: метрики качества, скорости и стоимости.

Соглашение о parse_error (фиксируется в отчёте):
* Accuracy считается по ВСЕМ N ответам, parse_error — неверный ответ;
* положительный класс — suspicious. TP: ожидали suspicious и получили suspicious;
  FN: ожидали suspicious, получили что угодно другое (включая parse_error);
  FP: ожидали normal, получили suspicious; TN: ожидали normal, получили normal;
* parse_error на примере с эталоном normal НЕ засчитывается как TN (нельзя «поощрять» сломанный
  формат), он влияет только на Accuracy и Format compliance.
"""
from __future__ import annotations

import difflib
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import ModelCfg, Settings
from parsing import PARSE_ERROR

POS = "suspicious"
NEG = "normal"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _counts(y_true, y_pred) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == POS) & (y_pred == POS)).sum())
    fn = int(((y_true == POS) & (y_pred != POS)).sum())
    fp = int(((y_true == NEG) & (y_pred == POS)).sum())
    tn = int(((y_true == NEG) & (y_pred == NEG)).sum())
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def classification_metrics(df: pd.DataFrame) -> dict:
    tp, fp, fn, tn = _counts(df["expected"], df["predicted"])
    n = len(df)
    precision, recall, f1 = _prf(tp, fp, fn)
    parse_errors = int((df["predicted"] == PARSE_ERROR).sum())
    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_errors": parse_errors,
        "format_compliance": _safe_div(int(df["format_ok"].sum()), n),
    }


def confusion(df: pd.DataFrame) -> dict:
    """Матрица ошибок: строки — эталон, столбцы — ответ модели (+ колонка parse_error)."""
    labels = [POS, NEG, PARSE_ERROR]
    matrix = confusion_matrix(df["expected"], df["predicted"], labels=labels)[:2]
    return {"rows": [POS, NEG], "cols": labels, "matrix": matrix.tolist()}


def bootstrap_ci(df: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """95 % доверительный интервал (перцентильный bootstrap) для Accuracy и F1."""
    rng = np.random.default_rng(seed)
    y_true = df["expected"].to_numpy()
    y_pred = df["predicted"].to_numpy()
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        tp, fp, fn, tn = _counts(y_true[idx], y_pred[idx])
        accs.append(_safe_div(tp + tn, len(idx)))
        f1s.append(_prf(tp, fp, fn)[2])
    lo, hi = 2.5, 97.5
    return {
        "accuracy_ci": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "n_boot": n_boot,
    }


def by_difficulty(df: pd.DataFrame) -> dict:
    out = {}
    for diff, part in df.groupby("difficulty"):
        out[str(diff)] = {
            "n": len(part),
            "accuracy": _safe_div(int((part["expected"] == part["predicted"]).sum()), len(part)),
            "errors": int((part["expected"] != part["predicted"]).sum()),
        }
    return out


def latency_and_cost(df: pd.DataFrame, model: ModelCfg) -> dict:
    lat = df["latency_s"].astype(float)
    tokens_in = pd.to_numeric(df["input_tokens"], errors="coerce")
    tokens_out = pd.to_numeric(df["output_tokens"], errors="coerce")
    have_usage = bool(tokens_in.notna().any())
    total_in = int(tokens_in.sum()) if have_usage else None
    total_out = int(tokens_out.sum()) if have_usage else None

    cost = None
    cost_per_success = None
    if have_usage and model.price_in is not None and model.price_out is not None:
        cost = (total_in * model.price_in + total_out * model.price_out) / 1_000_000
        successes = int((df["expected"] == df["predicted"]).sum())
        cost_per_success = _safe_div(cost, successes) if successes else None
    return {
        "latency_mean": float(lat.mean()),
        "latency_median": float(lat.median()),
        "latency_p95": float(lat.quantile(0.95)),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost": cost,
        "cost_per_success": cost_per_success,
    }


def high_confidence_errors(df: pd.DataFrame, threshold: float = 0.8) -> list[dict]:
    """Уверенные, но неверные ответы (вопрос 6 раздела 10)."""
    wrong = df[(df["expected"] != df["predicted"]) & (pd.to_numeric(df["confidence"], errors="coerce") >= threshold)]
    return wrong[["id", "expected", "predicted", "confidence", "reason"]].to_dict("records")


def repeat_stability(
    main: pd.DataFrame, rep: pd.DataFrame, judge: pd.DataFrame | None, tag: str
) -> dict:
    """Устойчивость: исходный ответ (запуск 0) + дополнительные запуски по выбранным примеру."""
    if rep is None or rep.empty:
        return {}
    base = main[main["id"].isin(rep["id"].unique())].copy()
    base["repeat"] = 0
    keep = ["id", "repeat", "predicted", "confidence", "reason", "format_ok", "expected"]
    allruns = pd.concat([base[keep], rep[keep]], ignore_index=True)

    per_example = []
    for row_id, g in allruns.groupby("id"):
        labels = g["predicted"].tolist()
        conf = pd.to_numeric(g["confidence"], errors="coerce")
        reasons = [str(r) for r in g["reason"].tolist() if str(r) and str(r) != "nan"]
        sims = [
            difflib.SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(reasons, 2)
        ]
        row = {
            "id": int(row_id),
            "runs": len(g),
            "labels": labels,
            "label_changed": len(set(labels)) > 1,
            "correct_runs": int((g["predicted"] == g["expected"]).sum()),
            "format_failures": int((~g["format_ok"].astype(bool)).sum()),
            "confidence_std": float(conf.std(ddof=0)) if conf.notna().sum() > 1 else None,
            "reason_similarity": float(np.mean(sims)) if sims else None,
            "judge_total_std": None,
        }
        if judge is not None and not judge.empty:
            j = judge[(judge["kind"] == "repeat") & (judge["model"] == tag.upper()) & (judge["id"] == row_id)]
            totals = pd.to_numeric(j["judge_total"], errors="coerce").dropna()
            if len(totals) > 1:
                row["judge_total_std"] = float(totals.std(ddof=0))
        per_example.append(row)

    def _mean(key: str):
        vals = [r[key] for r in per_example if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "examples": per_example,
        "n_examples": len(per_example),
        "label_changed_examples": sum(r["label_changed"] for r in per_example),
        "format_failures_total": sum(r["format_failures"] for r in per_example),
        "mean_confidence_std": _mean("confidence_std"),
        "mean_reason_similarity": _mean("reason_similarity"),
        "mean_judge_total_std": _mean("judge_total_std"),
    }


def compute_all(
    results: dict[str, pd.DataFrame],
    repeats: dict[str, pd.DataFrame],
    judge: pd.DataFrame | None,
    settings: Settings,
) -> dict:
    out: dict = {}
    for tag, df in results.items():
        model = settings.models[tag]
        out[tag] = {
            "model": model.name,
            **classification_metrics(df),
            **bootstrap_ci(df, settings.bootstrap_n),
            "confusion": confusion(df),
            "by_difficulty": by_difficulty(df),
            **latency_and_cost(df, model),
            "high_confidence_errors": high_confidence_errors(df),
            "stability": repeat_stability(df, repeats.get(tag), judge, tag),
        }
    return out
