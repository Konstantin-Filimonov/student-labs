"""Evaluation pipeline лабораторной работы № 4 (запуск одной командой).

    python src/pipeline.py                 # полный цикл: прогон → повторы → судья → метрики → отчёт
    python src/pipeline.py --mock          # отладка без реальных моделей (результаты не для отчёта!)
    python src/pipeline.py --stage report  # пересобрать отчёт (например, после ручной оценки)

Все артефакты запуска сохраняются в results/<run_id>/:
config.json, tests.csv, results_model_{a,b}.csv, repeats_model_{a,b}.csv, manual_eval.csv,
judge_scores.csv, metrics.json, summary_table.csv, report.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = ["run", "repeats", "judge", "metrics", "report"]


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Не сериализуется: {type(obj)}")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("reason", "format_error", "raw"):
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    df["format_ok"] = df["format_ok"].astype(bool)
    return df


def find_run_dir(results_dir: Path, run_id: str | None) -> Path:
    if run_id:
        path = results_dir / run_id
        if not path.exists():
            raise SystemExit(f"Папка запуска не найдена: {path}")
        return path
    runs = sorted(p for p in results_dir.iterdir() if p.is_dir() and (p / "config.json").exists())
    if not runs:
        raise SystemExit("В results/ нет запусков. Сначала выполните: python src/pipeline.py")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества LLM: evaluation pipeline")
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    parser.add_argument("--run-id", help="идентификатор запуска (папка в results/)")
    parser.add_argument("--tests", type=Path, help="CSV с тестовым набором (по умолчанию data/tests.csv)")
    parser.add_argument("--mock", action="store_true", help="имитация моделей для проверки конвейера")
    parser.add_argument("--force", action="store_true", help="перезаписать уже существующие результаты этапа")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK"] = "1"

    from benchmark import file_sha256, load_tests, run_model, run_repeats, select_hardest
    from config import RESULTS_DIR, TESTS_PATH, ModelCfg, Settings, load_settings
    from judge import build_manual_template, run_judge, select_sample_ids
    from metrics import compute_all
    from prompts import (
        JUDGE_SYSTEM_PROMPT,
        JUDGE_USER_TEMPLATE,
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        USER_TEMPLATE,
    )
    from report import (
        build_report,
        judge_summary,
        load_manual,
        manual_summary,
        save_summary_csv,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    stages = STAGES if args.stage == "all" else [args.stage]
    mock_flag = True if args.mock else None
    needs_api = any(s in stages for s in ("run", "repeats", "judge"))

    # ------------------------------------------------------------------ папка запуска
    creating = "run" in stages
    if creating and not args.run_id:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_mock = bool(mock_flag) or os.getenv("MOCK", "").lower() in {"1", "true", "yes"}
        run_dir = RESULTS_DIR / (f"{stamp}_{PROMPT_VERSION}" + ("_mock" if is_mock else ""))
    elif creating:
        run_dir = RESULTS_DIR / args.run_id
    else:
        run_dir = find_run_dir(RESULTS_DIR, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    tests_path = run_dir / "tests.csv"
    tests_src = args.tests or TESTS_PATH

    saved_meta = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None

    # Настройки: для обращений к API — из текущего .env (там ключи); для metrics/report —
    # из сохранённого config.json запуска (ключи не нужны, модели/параметры те же, что были)
    if needs_api or saved_meta is None:
        settings = load_settings(mock=mock_flag)
        if saved_meta and not creating:
            old = saved_meta["settings"]
            for key in ("model_a", "model_b", "judge"):
                if old[key]["name"] != getattr(settings, key).name:
                    print(f"ВНИМАНИЕ: {key} в .env ({getattr(settings, key).name}) отличается от "
                          f"использованной в запуске ({old[key]['name']})")
    else:
        saved = dict(saved_meta["settings"])
        for key in ("model_a", "model_b", "judge"):
            saved[key] = ModelCfg(**saved[key])
        settings = Settings(**saved)

    if saved_meta is not None:
        meta = saved_meta
    else:
        shutil.copyfile(tests_src, tests_path)
        tests_df = load_tests(tests_path)
        meta = {
            "run_id": run_dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_version": PROMPT_VERSION,
            "prompts": {
                "system": SYSTEM_PROMPT,
                "user_template": USER_TEMPLATE,
                "judge_system": JUDGE_SYSTEM_PROMPT,
                "judge_user_template": JUDGE_USER_TEMPLATE,
            },
            "tests_file": tests_src.name,
            "tests_sha256": file_sha256(tests_path),
            "n_tests": len(tests_df),
            "settings": settings.public(),
        }
        _write_json(cfg_path, meta)

    tests = load_tests(tests_path)
    if settings.mock:
        print("=" * 70 + "\n MOCK-РЕЖИМ: ответы имитируются, это НЕ результаты эксперимента\n" + "=" * 70)
    if settings.judge.name in {settings.model_a.name, settings.model_b.name}:
        print("ВНИМАНИЕ: судья совпадает с оцениваемой моделью — так делать нельзя (п. 4.8 методички).")
    print(f"Запуск: {run_dir.name}   этапы: {', '.join(stages)}\n")

    # ------------------------------------------------------------------ этап: прогон
    def results_path(tag: str) -> Path:
        return run_dir / f"results_model_{tag}.csv"

    def repeats_path(tag: str) -> Path:
        return run_dir / f"repeats_model_{tag}.csv"

    if "run" in stages:
        # модели идут строго друг за другом (сначала все запросы A, затем B): в LM Studio
        # это исключает перезагрузку модели между запросами и искажение latency
        for tag, model in settings.models.items():
            if results_path(tag).exists() and not args.force:
                print(f"[run] {results_path(tag).name} уже есть — пропуск (--force для перезаписи)")
                continue
            run_model(model, tests, settings, tag).to_csv(results_path(tag), index=False, encoding="utf-8-sig")

    def load_all(path_fn) -> dict[str, pd.DataFrame]:
        return {t: read_results(path_fn(t)) for t in settings.models if path_fn(t).exists()}

    results = load_all(results_path)
    if len(results) < 2 and any(s in stages for s in ("repeats", "judge", "metrics", "report")):
        raise SystemExit("Нет результатов основного прогона обеих моделей. Выполните этап run.")

    # ------------------------------------------------------------------ этап: повторы
    if "repeats" in stages:
        hard_ids = meta.get("hard_ids") or select_hardest(results, settings.n_hard)
        meta["hard_ids"] = hard_ids
        _write_json(cfg_path, meta)
        for tag, model in settings.models.items():
            if repeats_path(tag).exists() and not args.force:
                print(f"[repeats] {repeats_path(tag).name} уже есть — пропуск")
                continue
            run_repeats(model, tests, hard_ids, settings, tag).to_csv(
                repeats_path(tag), index=False, encoding="utf-8-sig"
            )
    repeats = load_all(repeats_path)

    # ------------------------------------------------------------------ этап: судья + ручной шаблон
    judge_path = run_dir / "judge_scores.csv"
    if "judge" in stages:
        sample_ids = meta.get("sample_ids") or select_sample_ids(tests, settings.manual_n, settings.sample_seed)
        meta["sample_ids"] = sample_ids
        _write_json(cfg_path, meta)
        build_manual_template(results, sample_ids, run_dir / "manual_eval.csv", tests)
        if judge_path.exists() and not args.force:
            print(f"[judge] {judge_path.name} уже есть — пропуск")
        else:
            run_judge(settings.judge, settings, tests, results, repeats, sample_ids).to_csv(
                judge_path, index=False, encoding="utf-8-sig"
            )
    judge_df = pd.read_csv(judge_path, encoding="utf-8-sig") if judge_path.exists() else None

    # ------------------------------------------------------------------ этапы: метрики и отчёт
    if "metrics" in stages or "report" in stages:
        metrics = compute_all(results, repeats, judge_df, settings)
        manual = load_manual(run_dir / "manual_eval.csv")
        manual_s = manual_summary(manual)
        _write_json(
            run_dir / "metrics.json",
            {"run_id": meta["run_id"], "models": metrics, "manual": manual_s, "judge": judge_summary(judge_df)},
        )
        save_summary_csv(run_dir / "summary_table.csv", metrics, manual_s, settings)
        print("[metrics] metrics.json, summary_table.csv сохранены")
        if "report" in stages:
            text = build_report(run_dir, meta, settings, metrics, results, manual, judge_df)
            (run_dir / "report.md").write_text(text, encoding="utf-8")
            print(f"[report] {run_dir / 'report.md'}")

    if "judge" in stages or args.stage == "all":
        print(
            "\nСледующие шаги:\n"
            f"  1. Заполните {run_dir / 'manual_eval.csv'} (колонки correctness, completeness, instruction, "
            "relevance — 0..2; hallucination — 0/1; comment) ДО просмотра judge_scores.csv;\n"
            f"  2. python src/pipeline.py --stage report --run-id {run_dir.name}"
        )


if __name__ == "__main__":
    main()
