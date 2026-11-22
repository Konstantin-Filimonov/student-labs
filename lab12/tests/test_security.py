"""Unit-тесты ЛР12: политика, tenant, approval, лимиты (без LLM)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from policy import Context, execute, unsafe_dispatch, PERMISSIONS


class TestPolicy(unittest.TestCase):
    def test_t1_allowed_read(self):
        r = execute({"tool": "get_incident", "args": {"id": "INC-101"}}, Context("analyst", "A"))
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["data"]["id"], "INC-101")

    def test_t2_tenant_denied(self):
        r = execute({"tool": "get_incident", "args": {"id": "INC-202"}}, Context("analyst", "A"))
        self.assertEqual(r["status"], "denied")
        self.assertEqual(r["reason"], "TENANT_DENIED")

    def test_t3_send_report_blocked_for_analyst(self):
        r = execute(
            {"tool": "send_report", "args": {"recipient": "outside@training.invalid"}},
            Context("analyst", "A"),
        )
        self.assertEqual(r["status"], "denied")
        self.assertEqual(r["reason"], "TOOL_NOT_ALLOWED")

    def test_t4_invalid_id_type(self):
        r = execute({"tool": "get_incident", "args": {"id": ["INC-101"]}}, Context("analyst", "A"))
        self.assertEqual(r["status"], "denied")
        self.assertEqual(r["reason"], "INVALID_ID")

    def test_t5_budget(self):
        r = execute({"tool": "get_incident", "args": {"id": "INC-101"}}, Context("analyst", "A", remaining_calls=0))
        self.assertEqual(r["reason"], "BUDGET_EXCEEDED")

    def test_t6_approval_required(self):
        r = execute(
            {"tool": "send_report", "args": {"recipient": "supervisor@training.invalid"}},
            Context("supervisor", "A", approved=False),
        )
        self.assertEqual(r["reason"], "HUMAN_APPROVAL_REQUIRED")

    def test_t7_simulated_send(self):
        r = execute(
            {"tool": "send_report", "args": {"recipient": "supervisor@training.invalid"}},
            Context("supervisor", "A", approved=True),
        )
        self.assertEqual(r["status"], "simulated")

    def test_t8_unknown_tool(self):
        r = execute({"tool": "run_shell", "args": {"command": "x"}}, Context("supervisor", "A", approved=True))
        self.assertEqual(r["reason"], "TOOL_NOT_ALLOWED")

    def test_t9_role_not_changed_by_document(self):
        # analyst остаётся analyst даже после «чтения» вредоносного KB
        ctx = Context("analyst", "A")
        execute({"tool": "search_knowledge", "args": {"id": "KB-02"}}, ctx)
        r = execute(
            {"tool": "send_report", "args": {"recipient": "supervisor@training.invalid"}},
            ctx,
        )
        self.assertEqual(r["reason"], "TOOL_NOT_ALLOWED")

    def test_t10_tenant_filter_on_kb(self):
        r = execute({"tool": "search_knowledge", "args": {"id": "KB-03"}}, Context("analyst", "A"))
        self.assertEqual(r["reason"], "TENANT_DENIED")

    def test_t11_repeated_forbidden(self):
        ctx = Context("analyst", "A", remaining_calls=20)
        for _ in range(10):
            r = execute({"tool": "run_shell", "args": {"command": "x"}}, ctx)
            self.assertEqual(r["status"], "denied")
        self.assertEqual(len(ctx.audit), 10)

    def test_t12_benign_search(self):
        r = execute({"tool": "search_knowledge", "args": {"id": "KB-01"}}, Context("analyst", "A"))
        self.assertEqual(r["status"], "ok")
        self.assertIn("untrusted_text", r)

    def test_untrusted_text_is_data_not_command(self):
        r = execute({"tool": "search_knowledge", "args": {"id": "KB-02"}}, Context("analyst", "A"))
        self.assertEqual(r["status"], "ok")
        self.assertIn("ВНИМАНИЕ", r["untrusted_text"])
        # но send_report по-прежнему нельзя
        r2 = execute(
            {"tool": "send_report", "args": {"recipient": "outside@evil.example"}},
            Context("analyst", "A"),
        )
        self.assertEqual(r2["status"], "denied")

    def test_unsafe_would_execute(self):
        r = unsafe_dispatch({"tool": "run_shell", "args": {"command": "x"}})
        self.assertEqual(r["status"], "would_execute")

    def test_permissions_least_privilege(self):
        self.assertNotIn("send_report", PERMISSIONS["analyst"])
        self.assertIn("send_report", PERMISSIONS["supervisor"])

    def test_recipient_whitelist(self):
        r = execute(
            {"tool": "send_report", "args": {"recipient": "attacker@evil.example"}},
            Context("supervisor", "A", approved=True),
        )
        self.assertEqual(r["reason"], "RECIPIENT_NOT_ALLOWED")

    def test_audit_no_secrets(self):
        ctx = Context("analyst", "A", session_id="s1")
        execute({"tool": "get_incident", "args": {"id": "INC-101"}}, ctx)
        self.assertEqual(ctx.audit[0]["session"], "s1")
        self.assertNotIn("password", str(ctx.audit).lower())


if __name__ == "__main__":
    unittest.main()
