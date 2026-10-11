"""Мониторинг ресурсов: CPU, RAM (psutil) и GPU/VRAM (nvidia-smi)."""
from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # не мигать консолью на Windows
_GB = 1024 ** 3


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    fallback = Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    return str(fallback) if fallback.exists() else None


def read_gpu() -> Optional[Dict[str, Any]]:
    """Возвращает данные первой NVIDIA-видеокарты или None, если GPU/nvidia-smi нет."""
    exe = _nvidia_smi()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        name = ",".join(parts[:-3])
        total, used, util = (float(x) for x in parts[-3:])
        return {"name": name, "vram_total_gb": total / 1024, "vram_used_gb": used / 1024,
                "util_pct": util}
    except Exception:
        return None


def sample_once(cpu_interval: Optional[float] = None) -> Dict[str, Any]:
    gpu = read_gpu()
    return {
        "cpu_pct": psutil.cpu_percent(interval=cpu_interval),
        "ram_gb": psutil.virtual_memory().used / _GB,
        "vram_gb": gpu["vram_used_gb"] if gpu else None,
        "gpu_pct": gpu["util_pct"] if gpu else None,
    }


class ResourceMonitor:
    """Фоновый сбор метрик с пометкой фазы эксперимента."""

    KEYS = ("cpu_pct", "ram_gb", "vram_gb", "gpu_pct")

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.phase = "init"
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self) -> None:
        psutil.cpu_percent(interval=None)  # «прогрев»: первый вызов всегда 0.0
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = sample_once()
            s["t_s"] = round(time.perf_counter() - self._t0, 2)
            s["phase"] = self.phase
            self.samples.append(s)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_phase(self, name: str) -> None:
        self.phase = name

    def summarize(self, phase: str) -> Dict[str, Any]:
        rows = [s for s in self.samples if s["phase"] == phase]
        result: Dict[str, Any] = {"phase": phase, "samples": len(rows)}
        for key in self.KEYS:
            vals = [r[key] for r in rows if r[key] is not None]
            result[key] = (
                {"mean": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)}
                if vals else None  # None = нет измерения (например, нет GPU)
            )
        return result

    def save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t_s", "phase", *self.KEYS])
            w.writeheader()
            for s in self.samples:
                w.writerow({k: s.get(k) for k in w.fieldnames})
