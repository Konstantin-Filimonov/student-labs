"""Инструменты (только чтение), JSON Schema, реестр REGISTRY и диспетчер dispatch.

Модель только ПРЕДЛАГАЕТ вызов; решение и выполнение — на стороне приложения:
имя из белого списка -> права роли -> JSON -> точный набор полей -> тип -> значение -> выполнение.
Функции никогда не вызываются через eval/exec/getattr: только через REGISTRY.
"""
import copy
import ipaddress
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
LIMITS = CFG["limits"]


def _load(name):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


INCIDENTS = _load("incidents.json")
IP_DATA = _load("ip_reputation.json")
ALERTS = _load("alerts.json")

# ---------- строгие валидаторы (ASCII-цифры: \d в Python принимает и «١٢٣») ----------
_INC = re.compile(r"INC-[0-9]{3}")
_ALR = re.compile(r"ALR-[0-9]{3}")


def v_incident_id(v):
    if not _INC.fullmatch(v):
        raise ValueError("Неверный идентификатор инцидента (формат INC-101)")
    return v


def v_alert_id(v):
    if not _ALR.fullmatch(v):
        raise ValueError("Неверный идентификатор оповещения (формат ALR-201)")
    return v


def v_ip(v):
    try:
        return ipaddress.ip_address(v).compressed
    except ValueError:
        raise ValueError("Некорректный IP-адрес") from None


def v_query(v):
    if not 1 <= len(v) <= LIMITS["max_query_chars"] or not v.strip():
        raise ValueError(f"Длина поискового запроса: 1–{LIMITS['max_query_chars']} символов")
    if any(ord(ch) < 32 for ch in v):
        raise ValueError("Недопустимые управляющие символы в запросе")
    return v


# ---------- сессия пользователя (роль и KB задаёт приложение, а не модель) ----------
@dataclass
class Session:
    role: str = field(default_factory=lambda: CFG["default_role"])
    kb: object = None

    @property
    def allowed_tools(self):
        return set(CFG["roles"].get(self.role, {"tools": []})["tools"])

    @property
    def kb_group(self):
        return CFG["roles"].get(self.role, {"kb_group": "public"})["kb_group"]


_default = None


def default_session():
    """Сессия по умолчанию: роль из config.yaml и детерминированный keyword-KB (для тестов без LLM)."""
    global _default
    if _default is None:
        from kb_search import KeywordKB
        _default = Session(kb=KeywordKB())
    return _default


# ---------- функции инструментов ----------
def get_incident(incident_id, session):
    return copy.deepcopy(INCIDENTS.get(incident_id, {"error": "not_found"}))


def lookup_ip(ip, session):
    return copy.deepcopy(IP_DATA.get(ip, {"risk": "unknown", "tag": "нет данных"}))


def get_alert(alert_id, session):
    return copy.deepcopy(ALERTS.get(alert_id, {"error": "not_found"}))


def search_knowledge_base(query, session):
    return session.kb.search(query, session.kb_group)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    param: str
    param_description: str
    validate: Callable
    fn: Callable


REGISTRY = {s.name: s for s in [
    ToolSpec("get_incident", "Прочитать учебный инцидент по ID", "incident_id", "Формат INC-101", v_incident_id, get_incident),
    ToolSpec("lookup_ip", "Получить учебную оценку риска IP-адреса", "ip", "IP-адрес IPv4 или IPv6", v_ip, lookup_ip),
    ToolSpec("search_knowledge_base", "Найти инструкции в учебной базе знаний (возвращает ID источника и текст)",
             "query", "Короткий поисковый запрос", v_query, search_knowledge_base),
    ToolSpec("get_alert", "Прочитать учебное оповещение SOC по ID", "alert_id", "Формат ALR-201", v_alert_id, get_alert),
]}


def tool_schemas(names=None):
    """JSON Schema инструментов в формате Chat Completions; схема генерируется из REGISTRY (не расходится с ним)."""
    out = []
    for name, s in REGISTRY.items():
        if names is not None and name not in names:
            continue
        out.append({"type": "function", "function": {
            "name": name, "description": s.description,
            "parameters": {"type": "object",
                           "properties": {s.param: {"type": "string", "description": s.param_description}},
                           "required": [s.param], "additionalProperties": False}}})
    return out


TOOLS = tool_schemas()


def dispatch(name, raw_arguments, session=None):
    """Единственная точка исполнения. Всегда возвращает словарь, не бросает исключений.

    {"error": "forbidden_tool"}      — имени нет в REGISTRY;
    {"error": "permission_denied"}   — инструмент есть, но роль пользователя его не допускает;
    {"error": "invalid_arguments"}   — некорректный JSON, набор полей, тип или значение;
    {"data": ...}                    — результат (включая {"error": "not_found"} внутри data);
    {"error": "internal_error"}      — сбой самой функции (ошибка приложения, не модели).
    """
    session = session or default_session()
    if not isinstance(name, str) or name not in REGISTRY:
        return {"error": "forbidden_tool"}
    if name not in session.allowed_tools:
        return {"error": "permission_denied"}
    spec = REGISTRY[name]
    try:
        if isinstance(raw_arguments, dict):  # часть серверов отдаёт аргументы уже словарём
            raw_arguments = json.dumps(raw_arguments)
        if not isinstance(raw_arguments, str) or len(raw_arguments) > LIMITS["max_raw_arguments_chars"]:
            raise ValueError("Аргументы должны быть JSON-строкой допустимой длины")
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict) or set(arguments) != {spec.param}:
            raise ValueError("Недопустимый набор параметров")
        value = arguments[spec.param]
        if not isinstance(value, str):
            raise ValueError("Параметр должен быть строкой")
        value = spec.validate(value)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        return {"error": "invalid_arguments", "details": str(exc)}
    try:
        return {"data": spec.fn(value, session)}
    except Exception:  # детали не передаём модели: это ошибка приложения
        return {"error": "internal_error"}
