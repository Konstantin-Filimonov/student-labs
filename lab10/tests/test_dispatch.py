"""Проверки диспетчера без LLM (раздел 5.2 методички + индивидуальное задание get_alert).
Запуск: py -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_search import KeywordKB  # noqa: E402
from tools import REGISTRY, TOOLS, Session, dispatch, tool_schemas  # noqa: E402


class MethodGuideAsserts(unittest.TestCase):
    """Дословные проверки из методических указаний."""

    def test_forbidden_tool(self):
        self.assertEqual(dispatch("delete_database", "{}"), {"error": "forbidden_tool"})

    def test_bad_ip(self):
        self.assertEqual(dispatch("lookup_ip", '{"ip":"bad-ip"}')["error"], "invalid_arguments")

    def test_extra_field(self):
        self.assertEqual(dispatch("get_incident", '{"incident_id":"INC-101","admin":true}')["error"], "invalid_arguments")

    def test_not_found(self):
        self.assertEqual(dispatch("get_incident", '{"incident_id":"INC-999"}'), {"data": {"error": "not_found"}})


class PositiveTests(unittest.TestCase):
    def test_get_incident(self):
        self.assertEqual(dispatch("get_incident", '{"incident_id":"INC-101"}')["data"]["severity"], "high")

    def test_lookup_ip_known_and_unknown(self):
        self.assertEqual(dispatch("lookup_ip", '{"ip":"203.0.113.42"}')["data"]["risk"], "high")
        self.assertEqual(dispatch("lookup_ip", '{"ip":"203.0.113.99"}')["data"]["risk"], "unknown")

    def test_lookup_ip_normalizes_ipv6(self):
        self.assertEqual(dispatch("lookup_ip", '{"ip":"2001:0db8:0000:0000:0000:0000:0000:0001"}')["data"]["risk"], "unknown")

    def test_search_returns_id_and_text(self):
        data = dispatch("search_knowledge_base", '{"query":"при high-инциденте"}')["data"]
        self.assertTrue(data and data[0]["id"] == "KB-01" and "SOC" in data[0]["text"])

    def test_result_is_a_copy(self):
        dispatch("get_incident", '{"incident_id":"INC-101"}')["data"]["severity"] = "hacked"
        self.assertEqual(dispatch("get_incident", '{"incident_id":"INC-101"}')["data"]["severity"], "high")


class NegativeTests(unittest.TestCase):
    def bad(self, name, raw):
        self.assertEqual(dispatch(name, raw)["error"], "invalid_arguments", (name, raw))

    def test_invalid_variants(self):
        self.bad("get_incident", "не json")
        self.bad("get_incident", "{}")
        self.bad("get_incident", '{"incident_id":101}')
        self.bad("get_incident", '[]')
        self.bad("get_incident", '{"incident_id":"INC-101\\n"}')   # fullmatch, а не $
        self.bad("get_incident", '{"incident_id":"INC-١٢٣"}')      # не-ASCII цифры
        self.bad("get_incident", '{"incident_id":"inc-101"}')
        self.bad("lookup_ip", '{"ip":"999.1.1.1"}')
        self.bad("lookup_ip", '{"ip":" 203.0.113.42"}')
        self.bad("search_knowledge_base", '{"query":""}')
        self.bad("search_knowledge_base", '{"query":"   "}')
        self.bad("search_knowledge_base", '{"query":"a\\u0000b"}')
        self.bad("search_knowledge_base", '{"query":"' + "а" * 181 + '"}')
        self.bad("get_incident", '{"incident_id":"INC-101"}' + " " * 3000)

    def test_unknown_and_nonstring_names(self):
        for name in ("os.system", "__import__", "GET_INCIDENT", "", None, 5):
            self.assertEqual(dispatch(name, "{}"), {"error": "forbidden_tool"})

    def test_role_permission_enforced_on_server(self):
        public = Session("public", KeywordKB())
        self.assertEqual(dispatch("get_incident", '{"incident_id":"INC-101"}', public), {"error": "permission_denied"})
        self.assertIn("data", dispatch("search_knowledge_base", '{"query":"журнал"}', public))
        self.assertEqual(dispatch("lookup_ip", '{"ip":"203.0.113.42"}', Session("unknown_role", KeywordKB())),
                         {"error": "permission_denied"})

    def test_internal_error_is_separate_and_not_leaked(self):
        class Broken:
            def search(self, q, g):
                raise RuntimeError("секрет внутри исключения")
        r = dispatch("search_knowledge_base", '{"query":"тест"}', Session("soc", Broken()))
        self.assertEqual(r, {"error": "internal_error"})


class AlertTool(unittest.TestCase):
    """Индивидуальное задание (вариант 1, SOC): get_alert — 4 позитивных и 2 негативных теста."""

    def test_pos_1(self):
        self.assertEqual(dispatch("get_alert", '{"alert_id":"ALR-201"}')["data"]["source"], "auth-service")

    def test_pos_2(self):
        self.assertEqual(dispatch("get_alert", '{"alert_id":"ALR-202"}')["data"]["severity"], "low")

    def test_pos_3_alert_without_incident(self):
        self.assertIsNone(dispatch("get_alert", '{"alert_id":"ALR-203"}')["data"]["incident_id"])

    def test_pos_4_not_found(self):
        self.assertEqual(dispatch("get_alert", '{"alert_id":"ALR-999"}'), {"data": {"error": "not_found"}})

    def test_neg_1_bad_format(self):
        self.assertEqual(dispatch("get_alert", '{"alert_id":"ALR-1"}')["error"], "invalid_arguments")

    def test_neg_2_wrong_field_name(self):
        self.assertEqual(dispatch("get_alert", '{"incident_id":"ALR-201"}')["error"], "invalid_arguments")


class SchemaTests(unittest.TestCase):
    def test_schema_matches_registry_and_is_strict(self):
        self.assertEqual({t["function"]["name"] for t in TOOLS}, set(REGISTRY))
        for t in TOOLS:
            p = t["function"]["parameters"]
            self.assertIs(p["additionalProperties"], False)
            self.assertEqual(p["required"], list(p["properties"]))

    def test_role_filtered_schemas(self):
        names = {t["function"]["name"] for t in tool_schemas({"search_knowledge_base"})}
        self.assertEqual(names, {"search_knowledge_base"})

    def test_all_tools_are_read_only_names(self):
        for name in REGISTRY:
            self.assertTrue(name.startswith(("get_", "lookup_", "search_")), name)


if __name__ == "__main__":
    unittest.main()
