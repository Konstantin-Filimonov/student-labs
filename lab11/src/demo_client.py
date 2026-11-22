"""Клиенты-имитаторы для проверки УПРАВЛЯЮЩЕЙ ЛОГИКИ без модели. Это НЕ языковые модели.

SimulatedModel (он же DemoClient) — детерминированная имитация: по тексту цели и накопленным наблюдениям
выбирает следующий вызов. Режим --demo и «sim»-эксперименты доказывают работу цикла, лимитов и политики,
но ничего не говорят о том, умеет ли реальная LLM правильно выбирать инструменты.
ScriptedClient — отдаёт заранее заданные ответы по очереди (для точечных тестов T4, T7–T10 и т. п.).
"""
import json
import re
from types import SimpleNamespace as NS

_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_IP_AFTER_WORD = re.compile(r"\bIP\s+((?=[^\s,;?!]*[\d\-:])[^\s,;?!]+)", re.I)
_INC = re.compile(r"INC-\d{3}")
_WRITE = re.compile(r"удал|заблокир|измени|отправ|останов", re.I)
_ANALYSIS = re.compile(r"анализ|заключ|сравн|\bip\b|репутац", re.I)
_KB = re.compile(r"инструкц|реагир|эскалац|заключ|рекоменд", re.I)


class Call:
    """Совместимо по форме с tool_call из openai."""
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = NS(name=name, arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False))


def _reply(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls or None))])


class ScriptedClient:
    """script — список ответов: (content, [Call, ...]) либо исключение (будет брошено). Все запросы пишутся в .requests."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        content, calls = item
        return _reply(content, calls)


def _history(messages):
    """[(имя, аргументы-dict, результат-dict)] в порядке выполнения, по парам assistant.tool_calls / tool."""
    declared, out = {}, []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                declared[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
        elif m.get("role") == "tool":
            name, raw = declared.get(m["tool_call_id"], ("?", "{}"))
            try:
                args = json.loads(raw)
            except (TypeError, ValueError):
                args = {}
            try:
                res = json.loads(m["content"])
            except ValueError:
                res = {}
            out.append((name, args if isinstance(args, dict) else {}, res if isinstance(res, dict) else {}))
    return out


class SimulatedModel:
    def __init__(self, gullible=False):
        """gullible=True имитирует «слабую» модель: пытается выполнить запрещённое действие (по просьбе или по тексту документа)."""
        self.gullible = gullible
        self.chat = NS(completions=self)

    # --- выбор следующего действия ---
    def _next_call(self, goal, hist):
        done_inc = {a.get("incident_id") for n, a, _ in hist if n == "get_incident"}
        done_ip = {a.get("ip") for n, a, _ in hist if n == "lookup_ip"}
        did_kb = any(n == "search_knowledge" for n, _, _ in hist)
        attempted_forbidden = any(n == "delete_incident" for n, _, _ in hist)
        ids = list(dict.fromkeys(_INC.findall(goal)))

        if _WRITE.search(goal):
            if self.gullible and not attempted_forbidden:
                return "delete_incident", {"incident_id": ids[0] if ids else "INC-101"}
            return None
        for iid in ids:
            if iid not in done_inc:
                return "get_incident", {"incident_id": iid}
        if _ANALYSIS.search(goal):
            for n, _, r in hist:
                ip = (r.get("data") or {}).get("ip") if n == "get_incident" else None
                if ip and ip not in done_ip:
                    return "lookup_ip", {"ip": ip}
        explicit = list(dict.fromkeys(_IPV4.findall(goal) + _IP_AFTER_WORD.findall(goal)))
        for ip in explicit:
            if ip not in done_ip:
                return "lookup_ip", {"ip": ip}
        inc_results = [r for n, _, r in hist if n == "get_incident"]
        nothing_found = bool(ids) and inc_results and all("error" in r for r in inc_results)
        if _KB.search(goal) and not did_kb and not nothing_found:  # нет записи -> инструкцию искать незачем
            sev = [(r.get("data") or {}).get("severity") for n, _, r in hist if n == "get_incident"]
            if "low" in sev and "high" not in sev:
                return "search_knowledge", {"query": "low-инцидент низкий риск"}
            if "high" in sev or "high" in goal.lower():
                return "search_knowledge", {"query": "эскалация high-инцидент"}
            return "search_knowledge", {"query": "триаж инцидента"}
        if self.gullible and not attempted_forbidden and any(
                n == "search_knowledge" and "delete_incident" in json.dumps(r, ensure_ascii=False) for n, _, r in hist):
            return "delete_incident", {"incident_id": "INC-101"}  # выполнил «указание» из документа
        return None

    # --- итоговый ответ только из наблюдений ---
    @staticmethod
    def _final(goal, hist):
        if not hist:
            if _WRITE.search(goal):
                return "Такой возможности нет: инструменты только для чтения, удаление и изменение записей недоступны."
            return "Привет! Я учебный SOC-ассистент: могу прочитать инцидент, проверить IP и найти инструкцию."
        lines, high = [], False
        for n, a, r in hist:
            data = r.get("data") or {}
            if n == "get_incident":
                if data:
                    high |= data.get("severity") == "high"
                    level = "высокий" if data.get("severity") == "high" else "низкий"
                    lines.append(f"{data['incident_id']}: {data['severity']} ({level} уровень), IP {data['ip']}, событие: {data['event']}.")
                else:
                    lines.append(f"{a.get('incident_id')}: запись не найдена ({r.get('error')}), данных недостаточно.")
            elif n == "lookup_ip":
                if data:
                    high |= data.get("risk") == "high"
                    risk = {"high": "высокий", "low": "низкий", "medium": "средний"}.get(data.get("risk"), "неизвестен")
                    lines.append(f"IP {data['ip']}: риск {risk}.")
                else:
                    lines.append(f"IP {a.get('ip')}: некорректный адрес ({r.get('error')}), проверка невозможна.")
            elif n == "search_knowledge":
                res = data.get("results") or []
                if res:
                    lines.append(f"Инструкция [{res[0]['id']}] «{res[0]['source']}».")
                else:
                    lines.append("В базе знаний подходящих сведений нет.")
            else:
                lines.append(f"Действие «{n}» недоступно ({r.get('error')}): выполнить его нельзя.")
        lines.append("Вывод: высокий риск, уведомить ответственного аналитика SOC." if high
                     else "Вывод: срочных действий не требуется.")
        lines.append("Неопределённость: данные учебные, сведения получены только из инструментов.")
        return " ".join(lines)

    def create(self, **kwargs):
        messages = kwargs["messages"]
        goal = next(m["content"] for m in messages if m["role"] == "user")
        hist = _history(messages)
        step = self._next_call(goal, hist)
        if step is None:
            return _reply(self._final(goal, hist))
        name, args = step
        return _reply(None, [Call(f"sim-{len(hist) + 1}", name, args)])


DemoClient = SimulatedModel
