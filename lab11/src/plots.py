"""Графики из experiments/metrics.csv: (1) успешность от max_rounds, (2) задержка и число вызовов.

  python src/plots.py
Строки mode=sim — проверка управляющей логики имитатором; mode=real — реальная LLM. Показываются раздельно.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent.parent / "experiments"
NAMES = {"sim": "Имитатор (не LLM)", "real": "Реальная LLM"}
STYLE = {"sim": ("#888888", "s", "--"), "real": ("#1f77b4", "o", "-")}


def main():
    path = EXP / "metrics.csv"
    if not path.exists():
        sys.exit("Нет experiments/metrics.csv — сначала запустите run_experiments.py")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    for m in by_mode:
        by_mode[m].sort(key=lambda r: int(r["max_rounds"]))
    (EXP / "plots").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, rs in by_mode.items():
        c, mk, ls = STYLE[m]
        ax.plot([int(r["max_rounds"]) for r in rs], [float(r["tsr_pct"]) for r in rs], color=c, marker=mk, ls=ls, label=NAMES[m])
    ax.set(xlabel="Ограничение числа раундов (max_rounds)", ylabel="Task Success Rate, %", ylim=(-3, 103),
           title="Успешность выполнения задач при разных лимитах раундов")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EXP / "plots" / "tsr_by_rounds.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for m, rs in by_mode.items():
        c, mk, ls = STYLE[m]
        x = [int(r["max_rounds"]) for r in rs]
        axes[0].plot(x, [float(r["latency_mean_s"]) for r in rs], color=c, marker=mk, ls=ls, label=NAMES[m])
        axes[1].plot(x, [float(r["mean_calls"]) for r in rs], color=c, marker=mk, ls=ls, label=NAMES[m])
    axes[0].set(xlabel="max_rounds", ylabel="Средняя задержка, с", title="Задержка")
    axes[1].set(xlabel="max_rounds", ylabel="Вызовов инструментов на задачу", title="Число вызовов")
    for a in axes:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(EXP / "plots" / "latency_calls.png", dpi=150)
    plt.close(fig)
    print("Графики сохранены в experiments/plots/")


if __name__ == "__main__":
    main()
