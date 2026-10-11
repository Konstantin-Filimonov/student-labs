"""Оркестратор лабораторной работы №5.

Запуск:  python src/run_all.py
Опции:   --model ID       другой идентификатор модели (задание 7, вторая модель)
         --out ПАПКА      куда писать результаты (по умолчанию results/)
         --skip STEP ...  пропустить шаги: bench dialog temperature log
         --no-pause       не ждать подтверждения выгрузки модели
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmark import resources_table, run_benchmark
from client import LMStudioClient, LMStudioError
from config import ROOT, load_config
from dialog import run_dialog
from log_analyzer import run_log_analysis
from monitor import ResourceMonitor
from plots import plot_benchmark, plot_resources
from prompts import PROMPT_VERSION
from report_data import build
from sysinfo import collect_system_info
from temperature import run_temperature


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # кириллица в консоли Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["bench", "dialog", "temperature", "log"])
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    client = LMStudioClient(cfg.base_url, args.model or cfg.model, cfg.timeout)
    try:
        client.model = client.resolve_model(args.model or cfg.model)
    except LMStudioError as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    print(f"Модель: {client.model}\nСервер: {cfg.base_url}\nРезультаты: {out}\n")

    print("[1/6] Сведения о компьютере...")
    sysinfo = collect_system_info(cfg.base_url)
    (out / "system_info.json").write_text(json.dumps(sysinfo, ensure_ascii=False, indent=2), encoding="utf-8")
    listen = sysinfo["api_server"]
    if listen.get("exposed_to_network"):
        print(f"  ВНИМАНИЕ: API слушает {listen['listen']} - доступен из сети! "
              "Отключите 'Serve on Local Network' в LM Studio.")

    data = {"started": started, "prompt_version": PROMPT_VERSION, "sysinfo": sysinfo,
            "dialog": [], "temperature": [], "log_analysis": {}}
    monitor = ResourceMonitor(cfg.sample_interval)

    if "bench" not in args.skip:
        print("\n[2/6] Подготовка к замеру 'до загрузки'.")
        print("  В LM Studio выгрузите модель (Eject / Unload), сервер оставьте запущенным.")
        print("  В настройках сервера должна быть включена загрузка модели по запросу (JIT loading).")
        if not args.no_pause:
            input("  Когда модель выгружена - нажмите Enter... ")
        state = (client.get_model_record() or {}).get("state", "unknown")
        cold_ok = state == "not-loaded"
        print(f"  Состояние модели: {state}" + ("" if cold_ok else "  (холодный старт не гарантирован!)"))

        monitor.start()
        monitor.set_phase("baseline")
        time.sleep(5)
        print("\n[3/6] Замер времени и скорости (задание 4) + ресурсы (задание 5)...")
        bench = run_benchmark(client, monitor, cfg.bench_runs, cfg.bench_max_tokens, cfg.temperature, out)
        monitor.stop()
        monitor.save_csv(out / "resources.csv")
        res = resources_table(monitor)
        (out / "resources_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "benchmark_summary.json").write_text(json.dumps(bench["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        plot_resources(monitor.samples, out / "resources.png")
        plot_benchmark(bench["rows"], out / "benchmark.png")
        record = client.get_model_record()
        data.update({"benchmark": bench, "resources": res, "cold_start_verified": cold_ok,
                     "model": {"id": client.model, "lmstudio_record": record,
                               "runtime": bench["summary"]["runtime"],
                               "model_info": bench["summary"]["model_info"],
                               "lmstudio_version_manual": cfg.lmstudio_version or "не указана (.env)"}})
        (out / "run_meta.json").write_text(json.dumps(data["model"], ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        data.update({"benchmark": {"rows": [], "summary": {}}, "resources": {}, "cold_start_verified": None,
                     "model": {"id": client.model}})

    if "dialog" not in args.skip:
        print("\n[4/6] Пять тестовых запросов (задание 2)...")
        data["dialog"] = run_dialog(client, cfg.temperature, out)
    if "temperature" not in args.skip:
        print("\n[5/6] Влияние температуры (задание 6)...")
        data["temperature"] = run_temperature(client, out)
    if "log" not in args.skip:
        print("\n[6/6] Индивидуальное задание: анализ журнала...")
        data["log_analysis"] = run_log_analysis(client, ROOT / "data" / "sample_auth.txt", out)

    if "bench" not in args.skip:
        (out / "report_data.md").write_text(build(data), encoding="utf-8")
        print(f"\nГОТОВО. Отправьте мне файл: {out / 'report_data.md'}")
        print("Не забудьте скриншоты (папка screenshots/): список моделей, запущенный сервер, мониторинг.")
    else:
        print("\nГОТОВО (бенчмарк пропущен, report_data.md не создавался).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
