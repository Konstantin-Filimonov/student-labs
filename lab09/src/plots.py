"""Графики (шаг 9): метрики от размера фрагмента и «качество — длина контекста».

Запуск: py src/plots.py -> results/plots/metric_by_chunk_size.png, results/plots/quality_vs_context.png
Если ручная разметка ещё не обобщена (review_summary.csv), второй график строится по MRR поиска.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ingest import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_chunk_size(summary):
    df = summary[(summary.search_mode == "dense") & (summary.rerank == "none")
                 & (summary.overlap == 100) & (summary.top_k == 4)].sort_values("chunk_size")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.chunk_size, df.hit_at_k, marker="o", label="Hit@4")
    ax.plot(df.chunk_size, df.mrr, marker="s", label="MRR")
    ax.set_xticks(df.chunk_size)
    ax.set_xlabel("Размер фрагмента, символов (overlap = 100, k = 4, dense)")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.05)
    ax.set_title("Качество поиска от размера фрагмента")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_by_chunk_size.png", dpi=200)
    plt.close(fig)


def plot_quality_vs_context(summary):
    review_path = RESULTS_DIR / "review_summary.csv"
    if review_path.exists():
        rv = pd.read_csv(review_path)
        df = rv.merge(summary[["experiment_id"]], on="experiment_id")
        x, y, ylabel = df.avg_context_chars, df.correctness_mean_0_2, "Средняя корректность ответа (0–2)"
        title = "Качество ответа и размер контекста"
    else:
        df = summary
        x, y, ylabel = df.avg_context_chars, df.mrr, "MRR (качество поиска; ответы не оценивались)"
        title = "Качество поиска и размер контекста"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y)
    for xi, yi, name in zip(x, y, df.experiment_id):
        ax.annotate(name, (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Средняя длина контекста, символов")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 2.05 if review_path.exists() else 1.05)  # полная шкала, чтобы не раздувать малые различия
    ax.set_xlim(0, max(x) * 1.25)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "quality_vs_context.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    plot_chunk_size(summary)
    plot_quality_vs_context(summary)
    print(f"Графики сохранены в {PLOTS_DIR}")
