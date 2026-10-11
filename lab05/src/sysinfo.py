"""Сбор сведений о компьютере (таблица 6.1 методических указаний)."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import psutil

from monitor import _NO_WINDOW, read_gpu

_GB = 1024 ** 3


def _cpu_name() -> str:
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW,
            )
            if out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "не определено"


def api_listen_address(base_url: str) -> Dict[str, Any]:
    """На каком адресе слушает локальный API (требование ИБ из раздела 5 методички)."""
    port = urlparse(base_url).port or 1234
    info: Dict[str, Any] = {"port": port, "listen": []}
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
                info["listen"].append(c.laddr.ip)
    except Exception as exc:  # на некоторых ОС нужны повышенные права
        info["error"] = str(exc)
    info["listen"] = sorted(set(info["listen"]))
    exposed = [ip for ip in info["listen"] if ip not in ("127.0.0.1", "::1")]
    info["exposed_to_network"] = bool(exposed) if info["listen"] else None
    return info


def collect_system_info(base_url: str, disk_path: Optional[str] = None) -> Dict[str, Any]:
    gpu = read_gpu()
    disk_root = disk_path or Path.cwd().anchor or "/"
    free_gb = shutil.disk_usage(disk_root).free / _GB
    return {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "threads_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / _GB, 1),
        "gpu": gpu["name"] if gpu else "нет (nvidia-smi не найден)",
        "vram_total_gb": round(gpu["vram_total_gb"], 1) if gpu else None,
        "disk_free_gb": round(free_gb, 1),
        "disk_checked": str(disk_root),
        "python": platform.python_version(),
        "api_server": api_listen_address(base_url),
    }
