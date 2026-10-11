"""Сборка отчёта запуска: report.md + summary_table.csv (раздел 9 методички)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Settings
from prompts import RUBRIC_CRITERIA

MANUAL_TO_JUDGE = {
    "correctness": "factual_correctness",
    "completeness": "completeness",
    "instruction": "instruction_following",
    "relevance": "relevance",
}


# ----------------------------- вспомогательные ------------------------------ #
def _pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}"


def _num(x, digits: int = 3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _int(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{int(x):,}".replace(",", " ")


def _to_bool(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in {"1", "1.0", "true", "да", "yes", "y"}:
        return True
    if s in {"0", "0.0", "false", "нет", "no", "n"}:
        return False
    return None


def _md_table(headers: list[str], rows: list[list]) -> str:
    def esc(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _best(a, b, higher: bool, eps: float = 1e-12) -> str:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "—"
    if abs(a - b) <= eps:
        return "равно"
    return "A" if (a > b) == higher else "B"


# ------------------------------ ручная оценка ------------------------------- #
def load_manual(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = list(MANUAL_TO_JUDGE)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["halluc_flag"] = df["hallucination"].map(_to_bool)
    df["filled"] = df[cols].notna().all(axis=1)
    df["manual_total"] = df[cols].sum(axis=1, min_count=4)
    return df


def manual_summary(manual: pd.DataFrame | None) -> dict:
    out: dict = {}
    if manual is None:
        return out
    for tag, g in manual.groupby("model"):
        g = g[g["filled"]]
        flags = g["halluc_flag"].dropna()
        out[str(tag).lower()] = {
            "n": len(g),
            "mean_total": float(g["manual_total"].mean()) if len(g) else None,
            "criteria": {c: float(g[c].mean()) for c in MANUAL_TO_JUDGE} if len(g) else {},
            "hallucination_rate": float(flags.mean()) if len(flags) else None,
            "n_halluc_marked": len(flags),
        }
    return out


def judge_summary(judge: pd.DataFrame | None) -> dict:
    out: dict = {}
    if judge is None or judge.empty:
        return out
    sample = judge[judge["kind"] == "sample"]
    for tag, g in sample.groupby("model"):
        totals = pd.to_numeric(g["judge_total"], errors="coerce")
        halluc = g["hallucination"].map(_to_bool).dropna()
        out[str(tag).lower()] = {
            "n": int(totals.notna().sum()),
            "mean_total": float(totals.mean()) if totals.notna().any() else None,
            "hallucination_rate": float(halluc.mean()) if len(halluc) else None,
            "errors": int((g["judge_error"].fillna("") != "").sum()),
        }
    return out


def agreement(manual: pd.DataFrame | None, judge: pd.DataFrame | None) -> dict:
    """Сравнение ручной оценки и LLM-as-a-Judge на одних и тех же парах (id, модель)."""
    if manual is None or judge is None or judge.empty:
        return {}
    m = manual[manual["filled"]].copy()
    if m.empty:
        return {}
    j = judge[judge["kind"] == "sample"].copy()
    j = j.rename(columns={**{c: f"j_{c}" for c in MANUAL_TO_JUDGE.values()}, "hallucination": "j_hallucination"})
    j = j[["id", "model", "judge_total"] + [f"j_{c}" for c in MANUAL_TO_JUDGE.values()] + ["j_hallucination"]]
    j["id"] = j["id"].astype(int)
    m["id"] = m["id"].astype(int)
    merged = m.merge(j, on=["id", "model"], how="inner")
    for c in [f"j_{v}" for v in MANUAL_TO_JUDGE.values()] + ["judge_total"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=["judge_total"])
    if merged.empty:
        return {}

    per_criterion = {}
    for mc, jc in MANUAL_TO_JUDGE.items():
        diff = (merged[mc] - merged[f"j_{jc}"]).abs()
        per_criterion[mc] = {
            "exact": float((diff == 0).mean()),
            "within1": float((diff <= 1).mean()),
            "mae": float(diff.mean()),
            "judge_minus_manual": float((merged[f"j_{jc}"] - merged[mc]).mean()),
        }
    total_diff = (merged["manual_total"] - merged["judge_total"]).abs()
    rho = None
    if len(merged) >= 3 and merged["manual_total"].nunique() > 1 and merged["judge_total"].nunique() > 1:
        rho = float(merged["manual_total"].rank().corr(merged["judge_total"].rank()))

    hpairs = merged.dropna(subset=["halluc_flag"]).copy()
    hpairs["judge_flag"] = hpairs["j_hallucination"].map(_to_bool)
    hpairs = hpairs.dropna(subset=["judge_flag"])
    return {
        "n_pairs": len(merged),
        "per_criterion": per_criterion,
        "total_mae": float(total_diff.mean()),
        "total_exact": float((total_diff == 0).mean()),
        "total_within1": float((total_diff <= 1).mean()),
        "spearman": rho,
        "halluc_agreement": float((hpairs["halluc_flag"] == hpairs["judge_flag"]).mean()) if len(hpairs) else None,
        "halluc_pairs": len(hpairs),
    }


# ------------------------------- сводка ------------------------------------- #
def summary_rows(metrics: dict, manual_s: dict, settings: Settings) -> list[list]:
    a, b = metrics["a"], metrics["b"]
    ma, mb = manual_s.get("a", {}), manual_s.get("b", {})
    cur = settings.price_currency

    def row(name, va, vb, fmt, higher):
        return [name, fmt(va), fmt(vb), _best(va, vb, higher)]

    rows = [
        row("Accuracy", a["accuracy"], b["accuracy"], _num, True),
        row("Precision (suspicious)", a["precision"], b["precision"], _num, True),
        row("Recall (suspicious)", a["recall"], b["recall"], _num, True),
        row("F1 (suspicious)", a["f1"], b["f1"], _num, True),
        row("Format compliance, %", a["format_compliance"], b["format_compliance"], _pct, True),
        row("Hallucination rate, % (ручная)", ma.get("hallucination_rate"), mb.get("hallucination_rate"), _pct, False),
        row("Средняя ручная оценка / 8", ma.get("mean_total"), mb.get("mean_total"), lambda x: _num(x, 2), True),
        row("Средняя latency, с", a["latency_mean"], b["latency_mean"], lambda x: _num(x, 2), False),
        row("Медианная latency, с", a["latency_median"], b["latency_median"], lambda x: _num(x, 2), False),
        row("p95 latency, с", a["latency_p95"], b["latency_p95"], lambda x: _num(x, 2), False),
        row("Входные токены", a["input_tokens"], b["input_tokens"], _int, False),
        row("Выходные токены", a["output_tokens"], b["output_tokens"], _int, False),
    ]
    if a["cost"] is None or b["cost"] is None:
        rows.append(["Оценочная стоимость теста", "не рассчитывалась", "не рассчитывалась", "—"])
    else:
        rows.append(row(f"Оценочная стоимость теста, {cur}", a["cost"], b["cost"], lambda x: _num(x, 4), False))
    return rows


def build_report(
    run_dir: Path,
    meta: dict,
    settings: Settings,
    metrics: dict,
    results: dict[str, pd.DataFrame],
    manual: pd.DataFrame | None,
    judge: pd.DataFrame | None,
) -> str:
    manual_s = manual_summary(manual)
    judge_s = judge_summary(judge)
    agree = agreement(manual, judge)
    a, b = metrics["a"], metrics["b"]
    L: list[str] = []

    L.append(f"# Отчёт запуска `{meta['run_id']}`\n")
    if settings.mock:
        L.append("> ⚠️ **MOCK-РЕЖИМ.** Ответы сгенерированы имитатором, а не реальными моделями. "
                 "Эти числа нельзя использовать в отчёте по лабораторной работе.\n")

    # 1. Конфигурация
    L.append("## 1. Конфигурация эксперимента\n")
    cfg_rows = [
        ["Дата и время запуска", meta["started_at"]],
        ["Версия промпта", meta["prompt_version"]],
        ["Тестовый набор", f"{meta['tests_file']} — {meta['n_tests']} примеров, sha256 `{meta['tests_sha256'][:16]}…`"],
        ["Модель A", f"`{settings.model_a.name}` ({settings.model_a.base_url})"],
        ["Модель B", f"`{settings.model_b.name}` ({settings.model_b.base_url})"],
        ["Судья (LLM-as-a-Judge)", f"`{settings.judge.name}` ({settings.judge.base_url}), temperature={settings.judge_temperature}"],
        ["Параметры генерации", f"temperature={settings.temperature}, max_tokens={settings.max_tokens}, seed={settings.seed}"],
        ["Structured Outputs (JSON Schema)", "включены" if settings.structured_output else "выключены (формат проверяется валидацией)"],
        ["Повторные прогоны", f"{settings.n_hard} самых сложных примеров × {settings.n_repeats} доп. запуска на модель"],
        ["Ручная оценка / судья", f"{settings.manual_n} ответов на модель (seed выборки {settings.sample_seed}, id: {meta.get('sample_ids')})"],
    ]
    if settings.judge.name in {settings.model_a.name, settings.model_b.name}:
        cfg_rows.append(["⚠️ Замечание", "судья совпадает с оцениваемой моделью — нарушение требования методички (п. 4.8)"])
    L.append(_md_table(["Параметр", "Значение"], cfg_rows) + "\n")

    # 2. Сводная таблица
    L.append("## 2. Сводная таблица результатов\n")
    L.append(_md_table(["Показатель", "Модель A", "Модель B", "Лучший результат"], summary_rows(metrics, manual_s, settings)) + "\n")
    L.append(f"Положительный класс — `suspicious`. Bootstrap 95 % ДИ ({a['n_boot']} ресэмплов):\n")
    L.append(_md_table(
        ["Модель", "Accuracy, 95 % ДИ", "F1, 95 % ДИ"],
        [[t.upper(), f"{_num(m['accuracy'])} [{_num(m['accuracy_ci'][0])}; {_num(m['accuracy_ci'][1])}]",
          f"{_num(m['f1'])} [{_num(m['f1_ci'][0])}; {_num(m['f1_ci'][1])}]"] for t, m in metrics.items()]) + "\n")
    if a["cost"] is None or b["cost"] is None:
        L.append("Стоимость не рассчитывалась (тариф не задан или API не вернул usage); показаны только токены.\n")
    else:
        L.append(f"Тариф: {settings.price_currency} за 1 млн токенов; источник: {settings.price_source or 'не указан'}; "
                 f"дата: {settings.price_date or 'не указана'}.\n")

    # 3. Матрицы ошибок
    L.append("## 3. Confusion matrix\n")
    for tag, m in metrics.items():
        c = m["confusion"]
        L.append(f"**Модель {tag.upper()}** (строки — эталон, столбцы — ответ модели; TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}, parse_error={m['parse_errors']})\n")
        L.append(_md_table(["эталон \\ ответ"] + c["cols"], [[r] + row for r, row in zip(c["rows"], c["matrix"])]) + "\n")

    # 4. Сложность
    L.append("## 4. Accuracy по уровням сложности\n")
    diffs = sorted(set(a["by_difficulty"]) | set(b["by_difficulty"]))
    L.append(_md_table(
        ["Сложность", "N", "A: accuracy (ошибок)", "B: accuracy (ошибок)"],
        [[d, a["by_difficulty"][d]["n"],
          f"{_num(a['by_difficulty'][d]['accuracy'], 2)} ({a['by_difficulty'][d]['errors']})",
          f"{_num(b['by_difficulty'][d]['accuracy'], 2)} ({b['by_difficulty'][d]['errors']})"] for d in diffs]) + "\n")

    # 5. Ошибки
    L.append("## 5. Ошибки моделей\n")
    for tag, df in results.items():
        wrong = df[df["expected"] != df["predicted"]].copy()
        L.append(f"**Модель {tag.upper()}**: ошибок {len(wrong)} из {len(df)}\n")
        if len(wrong):
            wrong["confidence"] = pd.to_numeric(wrong["confidence"], errors="coerce")
            wrong = wrong.sort_values("confidence", ascending=False, na_position="last")
            L.append(_md_table(
                ["id", "сложность", "эталон", "ответ", "conf", "объяснение / ошибка формата"],
                [[r["id"], r["difficulty"], r["expected"], r["predicted"], _num(r["confidence"], 2),
                  (r["reason"] if isinstance(r["reason"], str) and r["reason"] else r["format_error"])[:220]]
                 for _, r in wrong.iterrows()]) + "\n")
        hc = metrics[tag]["high_confidence_errors"]
        L.append(f"Уверенных (conf ≥ 0.8) ошибок: **{len(hc)}**.\n")

    # 6. Устойчивость
    L.append("## 6. Повторные прогоны (устойчивость)\n")
    stab_rows = []
    for tag, m in metrics.items():
        s = m["stability"]
        if s:
            stab_rows.append([tag.upper(), s["n_examples"], s["label_changed_examples"], s["format_failures_total"],
                              _num(s["mean_confidence_std"]), _num(s["mean_reason_similarity"], 2), _num(s["mean_judge_total_std"], 2)])
    if stab_rows:
        L.append(_md_table(
            ["Модель", "Примеров", "Сменили класс", "Нарушений формата", "Ср. σ confidence", "Ср. сходство объяснений (0–1)", "Ср. σ оценки судьи"],
            stab_rows) + "\n")
        for tag, m in metrics.items():
            s = m["stability"]
            if not s:
                continue
            L.append(f"**Модель {tag.upper()}** — по примерам:\n")
            L.append(_md_table(
                ["id", "ответы (запуск 0…N)", "класс менялся", "верных из N", "нарушений формата"],
                [[e["id"], ", ".join(e["labels"]), "да" if e["label_changed"] else "нет",
                  f"{e['correct_runs']}/{e['runs']}", e["format_failures"]] for e in s["examples"]]) + "\n")
    else:
        L.append("Повторные прогоны не выполнялись.\n")

    # 7. Ручная оценка
    L.append("## 7. Ручная оценка объяснений\n")
    if manual is None or not manual["filled"].any():
        L.append("Файл `manual_eval.csv` не заполнен — раздел будет рассчитан после заполнения "
                 "(`python src/pipeline.py --stage report`).\n")
    else:
        rows = []
        for tag, s in manual_s.items():
            crit = s["criteria"]
            rows.append([tag.upper(), s["n"], _num(crit.get("correctness"), 2), _num(crit.get("completeness"), 2),
                         _num(crit.get("instruction"), 2), _num(crit.get("relevance"), 2),
                         _num(s["mean_total"], 2), f"{_pct(s['hallucination_rate'])} ({s['n_halluc_marked']} отв.)"])
        L.append(_md_table(["Модель", "N", "Correctness", "Completeness", "Instruction", "Relevance", "Итого /8", "Hallucination rate, %"], rows) + "\n")

    # 8. Судья
    L.append("## 8. LLM-as-a-Judge и сравнение с ручной оценкой\n")
    if not judge_s:
        L.append("Судья не запускался.\n")
    else:
        L.append(_md_table(
            ["Модель", "Оценено", "Ср. оценка судьи /8", "Hallucination (судья), %", "Ошибок судьи"],
            [[t.upper(), s["n"], _num(s["mean_total"], 2), _pct(s["hallucination_rate"]), s["errors"]] for t, s in judge_s.items()]) + "\n")
    if agree:
        L.append(f"Сопоставлено пар (ручная оценка ↔ судья): **{agree['n_pairs']}**.\n")
        L.append(_md_table(
            ["Критерий", "Точное совпадение, %", "Расхождение ≤ 1, %", "MAE", "Судья − человек (смещение)"],
            [[c, _pct(v["exact"]), _pct(v["within1"]), _num(v["mae"], 2), f"{v['judge_minus_manual']:+.2f}"]
             for c, v in agree["per_criterion"].items()]) + "\n")
        L.append(f"Суммарная оценка /8: MAE = {_num(agree['total_mae'], 2)}, точное совпадение = {_pct(agree['total_exact'])} %, "
                 f"расхождение ≤ 1 = {_pct(agree['total_within1'])} %, ранговая корреляция Спирмена = {_num(agree['spearman'], 2)}. "
                 f"Совпадение флага галлюцинации: {_pct(agree['halluc_agreement'])} % ({agree['halluc_pairs']} пар).\n")
    elif judge_s:
        L.append("Сравнение с ручной оценкой будет доступно после заполнения `manual_eval.csv`.\n")

    L.append("## 9. Артефакты запуска\n")
    L.append("`config.json`, `tests.csv`, `results_model_a.csv`, `results_model_b.csv`, `repeats_model_*.csv`, "
             "`manual_eval.csv`, `judge_scores.csv`, `metrics.json`, `summary_table.csv`, `report.md`.\n")
    return "\n".join(L)


def save_summary_csv(path: Path, metrics: dict, manual_s: dict, settings: Settings) -> None:
    rows = summary_rows(metrics, manual_s, settings)
    pd.DataFrame(rows, columns=["Показатель", "Модель A", "Модель B", "Лучший результат"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
