"""Поддельный клиент Chat Completions для тестов протокола без реальной модели."""
import json
from types import SimpleNamespace


class FakeCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments if isinstance(arguments, str) else json.dumps(arguments))

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


class FakeClient:
    """script — список ответов по очереди: (content, [FakeCall...]). Записывает все запросы в .requests."""

    def __init__(self, script, fail_tool_choice_none=False):
        self.script, self.requests, self.fail = list(script), [], fail_tool_choice_none
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail and kwargs.get("tool_choice") == "none":
            raise RuntimeError("tool_choice=none не поддерживается")
        content, calls = self.script.pop(0)
        msg = SimpleNamespace(content=content, tool_calls=calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
