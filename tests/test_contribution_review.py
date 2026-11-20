"""Offline tests for review evidence; these do not validate the reference labels."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.prepare_contribution_review import build_review_packet, parse_event_log


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_contribution_review.py"
FIXTURES = ROOT / "tests" / "fixtures" / "contribution_review"


class ContributionReviewTest(unittest.TestCase):
    def fixture(self, name):
        return parse_event_log((FIXTURES / name).read_text(encoding="utf-8"))

    def test_calibration_cases_have_resolvable_reference_evidence(self):
        references = json.loads(
            (FIXTURES / "reference_annotations.json").read_text(encoding="utf-8")
        )["cases"]
        self.assertEqual(set(references), {p.name for p in FIXTURES.glob("*.jsonl")})
        self.assertEqual(len(references), 8)
        for name, reference in references.items():
            with self.subTest(case=name):
                events = self.fixture(name)
                packet = build_review_packet(events, ["user"])
                anchor = reference["contribution_event_index"]
                self.assertEqual(
                    packet["reviews"][0]["contribution_event_index"], anchor
                )
                self.assertIn(reference["contribution_span"], events[anchor]["action"])
                for index in reference["evidence_event_indices"]:
                    self.assertGreaterEqual(index, anchor)
                    self.assertLess(index, len(events))
                # Reference judgments must never leak into a reviewer's packet.
                for review in packet["reviews"]:
                    self.assertIsNone(review["disposition"])
                    self.assertIsNone(review["contribution_span"])
                    self.assertIsNone(review["explanation"])
                    self.assertEqual(review["evidence_event_indices"], [])

    def test_matched_criterion_exposes_different_final_work(self):
        applied = build_review_packet(self.fixture("01_implemented.jsonl"), ["user"])
        ignored = build_review_packet(
            self.fixture("02_acknowledged_but_ignored.jsonl"), ["user"]
        )
        self.assertEqual(applied["events"][:2], ignored["events"][:2])
        self.assertNotEqual(
            applied["events"][-1]["public_observation"],
            ignored["events"][-1]["public_observation"],
        )

    def test_preserves_failed_edits_and_later_observed_state(self):
        events = self.fixture("07_failed_edit.jsonl")
        packet = build_review_packet(events, ["user"])
        self.assertEqual(len(packet["events"]), len(events))
        self.assertTrue(packet["events"][2]["action_status"].startswith("failed"))
        self.assertIn("Retain the announcement", packet["events"][2]["action"])
        self.assertEqual(
            packet["events"][0]["public_observation"],
            packet["events"][-1]["public_observation"],
        )

    def test_human_workspace_edits_are_reviewable_and_attributable(self):
        packet = build_review_packet(
            self.fixture("08_human_edit_overwritten.jsonl"), ["user"]
        )
        self.assertEqual(packet["reviews"][0]["contribution_event_index"], 1)
        self.assertEqual(packet["events"][1]["action_type"], "environment")
        self.assertEqual(packet["events"][1]["role"], "user")
        self.assertNotEqual(
            packet["events"][1]["public_observation"],
            packet["events"][-1]["public_observation"],
        )

    def test_exact_roles_and_statuses_without_substring_matching(self):
        original = self.fixture("05_routine_closure.jsonl")[1]
        events = [
            dict(original, role="user_simulator_service"),
            dict(original, role="participant", action_status="failed"),
            dict(original, role="participant"),
            dict(original, role="second_person"),
        ]
        packet = build_review_packet(events, ["participant", "second_person"])
        self.assertEqual(
            [r["contribution_event_index"] for r in packet["reviews"]], [2, 3]
        )
        self.assertEqual(len(packet["events"]), 4)
        with self.assertRaisesRegex(ValueError, "role names"):
            build_review_packet(events, ["user"])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_review_packet(events, [])
        with self.assertRaisesRegex(ValueError, "collection"):
            build_review_packet(events, "user")

    def test_omits_private_observations_and_preserves_missingness(self):
        event = self.fixture("01_implemented.jsonl")[1]
        event["current_observation"]["private"] = {"user": {"secret": "PRIVATE"}}
        event["current_chat_history"] = [{"private": "PRIVATE"}]
        packet = build_review_packet([event], ["user"])
        self.assertNotIn("PRIVATE", json.dumps(packet))
        packet["events"][0]["public_observation"]["result_editor"] = "changed"
        self.assertNotEqual(
            event["current_observation"]["public"]["result_editor"], "changed"
        )
        del event["current_observation"]
        packet = build_review_packet([event], ["user"])
        self.assertIsNone(packet["events"][0]["public_observation"])

    def test_truncated_log_is_retained_without_automatic_failure_label(self):
        events = self.fixture("06_incomplete_record.jsonl")
        packet = build_review_packet(events, ["user"])
        self.assertEqual(packet["event_count"], 3)
        self.assertEqual(packet["events"][-1]["action"], events[-1]["action"])
        self.assertIsNone(packet["reviews"][0]["disposition"])

    def test_invalid_json_and_incomplete_events_report_physical_line(self):
        for text in ["\n{broken", '\n{"role":"user"}', '\n[]']:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "Line 2"):
                    parse_event_log(text)

    def test_cli_preserves_unicode_newlines_and_source_identity(self):
        event = self.fixture("01_implemented.jsonl")[1]
        event["action"] = "SEND_TEAMMATE_MESSAGE(message=naïve café\n保留原意)"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "log.jsonl"
            target = Path(directory) / "review.json"
            raw = ("\n" + json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            source.write_bytes(raw)
            command = [
                sys.executable, str(SCRIPT), "--event-log", str(source),
                "--human-role", "user", "--output", str(target),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            packet = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(packet["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(packet["events"][0]["action"], event["action"])
            self.assertEqual(packet["reviews"][0]["contribution_event_index"], 0)
            prior = target.read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(target.read_bytes(), prior)
            command[-1] = str(source)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(source.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
