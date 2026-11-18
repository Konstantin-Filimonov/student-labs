"""Протокол tool_calls / tool_call_id, лимиты и фазы — на поддельном клиенте."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llm import FakeCall, FakeClient  # noqa: E402
from kb_search import KeywordKB  # noqa: E402
from tool_loop import run_no_tools, run_once  # noqa: E402

SOC = lambda: __import__("tools").Session("soc", KeywordKB())  # noqa: E731
PUB = lambda: __import__("tools").Session("public", KeywordKB())  # noqa: E731


class ToolLoopTests(unittest.TestCase):
    def test_no_tool_call_single_request(self):
        c = FakeClient([("Привет!", None)])
        out = run_once("Привет!", c, "m", SOC())
        self.assertEqual((out["answer"], out["tool_calls"], len(c.requests)), ("Привет!", 0, 1))
        self.assertEqual(c.requests[0]["tool_choice"], "auto")

    def test_full_cycle_and_protocol(self):
        c = FakeClient([(None, [FakeCall("call_1", "get_incident", {"incident_id": "INC-101"})]), ("Высокий.", None)])
        out = run_once("INC-101?", c, "m", SOC(), keep_messages=True)
        roles = [m["role"] for m in out["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        tool_msg = out["messages"][3]
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertEqual(out["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(c.requests[1]["tool_choice"], "none")  # новые вызовы в фазе ответа запрещены
        self.assertTrue(out["trace"][0]["executed"])

    def test_unknown_tool_is_blocked_but_answered(self):
        c = FakeClient([(None, [FakeCall("c1", "delete_database", "{}")]), ("Не могу.", None)])
        out = run_once("Удали всё", c, "m", SOC(), keep_messages=True)
        self.assertEqual(out["trace"][0]["error"], "forbidden_tool")
        self.assertFalse(out["trace"][0]["executed"])
        self.assertEqual(out["messages"][3]["tool_call_id"], "c1")

    def test_limit_three_executed_every_call_answered(self):
        calls = [FakeCall(f"c{i}", "get_incident", {"incident_id": "INC-101"}) for i in range(5)]
        c = FakeClient([(None, calls), ("Готово", None)])
        out = run_once("много", c, "m", SOC(), keep_messages=True)
        self.assertEqual([t["error"] for t in out["trace"]], [None, None, None, "limit_exceeded", "limit_exceeded"])
        self.assertEqual({m["tool_call_id"] for m in out["messages"] if m["role"] == "tool"}, {f"c{i}" for i in range(5)})

    def test_second_phase_tool_calls_are_not_executed(self):
        c = FakeClient([(None, [FakeCall("c1", "get_incident", {"incident_id": "INC-102"})]),
                        (None, [FakeCall("c2", "delete_database", "{}")])])
        out = run_once("x", c, "m", SOC())
        self.assertEqual(out["second_phase_tool_calls_ignored"], 1)
        self.assertEqual(len(out["trace"]), 1)
        self.assertIn("вместо ответа", out["answer"])

    def test_tool_choice_none_fallback(self):
        c = FakeClient([(None, [FakeCall("c1", "get_incident", {"incident_id": "INC-102"})]), ("низкий", None)],
                       fail_tool_choice_none=True)
        out = run_once("x", c, "m", SOC())
        self.assertTrue(out["second_phase_fallback"])
        self.assertEqual(out["answer"], "низкий")

    def test_role_limits_tools_offered_to_model(self):
        c = FakeClient([("ок", None)])
        run_once("x", c, "m", PUB())
        offered = {t["function"]["name"] for t in c.requests[0]["tools"]}
        self.assertEqual(offered, {"search_knowledge_base"})

    def test_hallucinated_permitted_name_is_denied_for_public_role(self):
        c = FakeClient([(None, [FakeCall("c1", "get_incident", {"incident_id": "INC-101"})]), ("нет доступа", None)])
        out = run_once("INC-101", c, "m", PUB())
        self.assertEqual(out["trace"][0]["error"], "permission_denied")

    def test_injection_text_in_tool_result_cannot_trigger_execution(self):
        from kb_search import read_jsonl
        from tools import BASE_DIR, Session
        poisoned = KeywordKB(read_jsonl(BASE_DIR / "data" / "kb_poisoned.jsonl"))
        c = FakeClient([(None, [FakeCall("c1", "search_knowledge_base", {"query": "high-инцидент"})]),
                        (None, [FakeCall("c2", "delete_database", "{}")])])
        out = run_once("инструкция", c, "m", Session("soc", poisoned), keep_messages=True)
        self.assertIn("PWNED-1337", out["messages"][3]["content"])  # инъекция дошла до модели как ДАННЫЕ
        self.assertEqual(len(out["trace"]), 1)                       # но новый вызов не исполнен

    def test_no_tools_mode_sends_no_tools(self):
        c = FakeClient([("Ответ", None)])
        out = run_no_tools("вопрос", c, "m")
        self.assertNotIn("tools", c.requests[0])
        self.assertEqual(out["mode"], "no_tools")


if __name__ == "__main__":
    unittest.main()
