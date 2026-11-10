"""Сводка по ручной разметке ответов (вторая таблица отчёта).

Заполните в results/answer_review.csv колонки:
  correctness_0_2     — 0 / 1 / 2 (только для ответимых вопросов)
  grounded_yes_no     — да / нет (утверждения следуют из найденных фрагментов)
  citation_ok_yes_no  — да / нет (ссылка [S#] действительно подтверждает утверждение)
  refusal_ok_yes_no   — да / нет (только для неответимых: модель корректно отказалась)
Запуск: py src/summarize_review.py -> results/review_summary.csv
"""
import pandas as pd

from ingest import RESULTS_DIR

YES = {"да", "yes", "y", "1", "true", "д"}
NO = {"нет", "no", "n", "0", "false", "н"}


def to_bin(value):
    s = str(value).strip().lower()
    if s in YES:
        return 1.0
    if s in NO:
        return 0.0
    return float("nan")


def summarize(review_path=RESULTS_DIR / "answer_review.csv"):
    df = pd.read_csv(review_path, encoding="utf-8-sig")
    for c in ("grounded_yes_no", "citation_ok_yes_no", "refusal_ok_yes_no"):
        df[c] = df[c].map(to_bin)
    df["correctness_0_2"] = pd.to_numeric(df["correctness_0_2"], errors="coerce")
    rows = []
    for exp_id, g in df.groupby("experiment_id", sort=False):
        ans, no = g[g.answerable], g[~g.answerable]
        rows.append({"experiment_id": exp_id,
                     "n_answerable_rated": int(ans.correctness_0_2.notna().sum()),
                     "correctness_mean_0_2": round(ans.correctness_0_2.mean(), 3),
                     "grounded_pct": round(100 * ans.grounded_yes_no.mean(), 1),
                     "citation_ok_pct": round(100 * ans.citation_ok_yes_no.mean(), 1),
                     "n_unanswerable_rated": int(no.refusal_ok_yes_no.notna().sum()),
                     "refusal_ok_pct": round(100 * no.refusal_ok_yes_no.mean(), 1),
                     "auto_has_citation_pct": round(100 * ans.has_citation.mean(), 1),
                     "avg_context_chars": round(g.context_chars.mean(), 1),
                     "avg_t_generate_s": round(g.t_generate_s.mean(), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "review_summary.csv", index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    summarize()
