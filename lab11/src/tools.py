"""Инструменты агента (ТОЛЬКО ЧТЕНИЕ) и диспетчер с независимой от LLM политикой.

Модель лишь ПРЕДЛАГАЕТ вызов. Решение принимает приложение, по цепочке:
имя из белого списка -> права роли -> размер/JSON -> точный набор полей -> тип -> значение -> выполнение.
Функции вызываются только через REGISTRY (никаких eval/exec/getattr/shell/URL).

Коды результата dispatch (всегда словарь, исключения наружу не выходят):
  {"data": ...}                   — успешное выполнение;
  {"error": "forbidden_tool"}     — имени нет в REGISTRY (например, delete_incident);
  {"error": "permission_denied"}  — инструмент существует, но роль его не допускает;
  {"error": "invalid_json"}       — аргументы не являются корректным JSON-объектом;
  {"error": "invalid_arguments"}  — лишние/отсутствующие поля, неверный тип, слишком длинное значение;
  {"error": "invalid_id"}         — неверный формат идентификатора инцидента;
  {"error": "invalid_ip"}         — не IP-адрес;
  {"error": "invalid_query"}      — недопустимый поисковый запрос;
  {"error": "not_found"}          — корректный запрос, но записи нет (внутри функции инструмента);
  {"error": "internal_error"}     — сбой самой функции (ошибка приложения, а не модели).
"""
import copy
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MAX_RAW_ARGS_CHARS = 2000
MAX_VALUE_CHARS = 200
MAX_QUERY_CHARS = 180

# Права задаёт приложение, а не системный промпт.
ROLE_TOOLS = {
    "soc": frozenset({"get_incident", "lookup_ip", "search_knowledge"}),
    "public": frozenset({"search_knowledge"}),
}
ROLE_KB_GROUP = {"soc": "soc", "public": "public"}  # soc видит public + soc (ACL как в ЛР9)

INVALID_CODES = frozenset({"forbidden_tool", "permission_denied", "invalid_json", "invalid_arguments",
                           "invalid_id", "invalid_ip", "invalid_query"})


class ToolError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _load(name):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


INCIDENTS = _load("incidents.json")
IP_DATA = _load("ip_reputation.json")

# ASCII-цифры: \d в Python принимает и «١٢٣», поэтому [0-9]
_INC = re.compile(r"INC-[0-9]{3}")


def v_incident_id(v):
    if not _INC.fullmatch(v):
        raise ToolError("invalid_id")
    return v


def v_ip(v):
    try:
        return ipaddress.ip_address(v).compressed
    except ValueError:
        raise ToolError("invalid_ip") from None


def v_query(v):
    if not v.strip() or len(v) > MAX_QUERY_CHARS or any(ord(ch) < 32 for ch in v):
        raise ToolError("invalid_query")
    return v


@dataclass
class Session:
    """Контекст пользователя. Роль и база знаний задаются приложением; модель на них не влияет."""
    role: str = "soc"
    kb: object = None

    @property
    def allowed_tools(self):
        return ROLE_TOOLS.get(self.role, frozenset())

    @property
    def kb_group(self):
        return ROLE_KB_GROUP.get(self.role, "public")


def default_session(role="soc"):
    from kb import KeywordKB
    return Session(role=role, kb=KeywordKB())


# ---------- функции инструментов ----------
def get_incident(incident_id, session):
    rec = INCIDENTS.get(incident_id)
    if rec is None:
        return {"error": "not_found"}
    return {"incident_id": incident_id, **copy.deepcopy(rec)}


def lookup_ip(ip, session):
    rec = IP_DATA.get(ip)
    return {"ip": ip, **copy.deepcopy(rec)} if rec else {"ip": ip, "risk": "unknown", "tag": "нет данных в учебной базе"}


def search_knowledge(query, session):
    return {"results": session.kb.search(query, session.kb_group)}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    param: str
    param_description: str
    validate: Callable
    fn: Callable


REGISTRY = {s.name: s for s in [
    ToolSpec("get_incident", "Прочитать учебный инцидент SOC по ID (возвращает severity, ip, event)",
             "incident_id", "Формат INC-101", v_incident_id, get_incident),
    ToolSpec("lookup_ip", "Получить учебную оценку риска IP-адреса (IPv4 или IPv6)",
             "ip", "IP-адрес, например 203.0.113.42", v_ip, lookup_ip),
    ToolSpec("search_knowledge", "Найти инструкции SOC в учебной базе знаний (возвращает ID источника и текст)",
             "query", "Короткий поисковый запрос", v_query, search_knowledge),
]}


def tool_schemas(names=None):
    """JSON Schema в формате Chat Completions; генерируется из REGISTRY, поэтому не расходится с ним."""
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


def normalize_raw(raw):
    """Некоторые серверы отдают аргументы словарём; приводим к строке JSON (канонический вид для детекции повторов)."""
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return raw


def dispatch(name, raw_arguments, session=None):
    """Единственная точка исполнения инструментов. Не бросает исключений."""
    session = session or default_session()
    if not isinstance(name, str) or name not in REGISTRY:
        return {"error": "forbidden_tool"}
    if name not in session.allowed_tools:
        return {"error": "permission_denied"}
    spec = REGISTRY[name]
    raw_arguments = normalize_raw(raw_arguments)
    if not isinstance(raw_arguments, str):
        return {"error": "invalid_json"}
    if len(raw_arguments) > MAX_RAW_ARGS_CHARS:
        return {"error": "invalid_arguments"}
    try:
        args = json.loads(raw_arguments)
    except (ValueError, RecursionError):
        return {"error": "invalid_json"}
    if not isinstance(args, dict) or set(args) != {spec.param}:
        return {"error": "invalid_arguments"}
    value = args[spec.param]
    if type(value) is not str or len(value) > MAX_VALUE_CHARS:
        return {"error": "invalid_arguments"}
    try:
        value = spec.validate(value)
    except ToolError as exc:
        return {"error": exc.code}
    try:
        result = spec.fn(value, session)
    except Exception:  # детали модели не передаём: это ошибка приложения
        return {"error": "internal_error"}
    return result if "error" in result else {"data": result}
