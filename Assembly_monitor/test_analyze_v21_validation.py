"""Smoke tests for the corrected V2.1 validation analyser.

Run with: python -m unittest test_analyze_v21_validation.py
These tests validate analyser logic only; they are not evidence of real-camera system accuracy.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("analyze_v21_validation.py")
spec = importlib.util.spec_from_file_location("analyze_v21_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationAnalyserTests(unittest.TestCase):
    def test_csv_notes_with_commas_are_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trials.csv"
            p.write_text(
                "# comment line\n"
                "log_file,trial_id,condition,true_stages_completed,notes\n"
                'a.jsonl,T2-01,nominal,7,"hands occluded, brief pause"\n',
                encoding="utf-8",
            )
            rows = module.read_trials(p)
            self.assertEqual(rows["a.jsonl"]["notes"], "hands occluded, brief pause")

    def test_error_latency_is_relative_to_injection_time(self):
        session = {"warnings": [
            {"warning_type": "WRONG_PART", "elapsed_s": 12.0},  # before error
            {"warning_type": "WRONG_PART", "elapsed_s": 18.5},  # after error
        ]}
        self.assertEqual(module.error_response_time(session, "WRONG_PART", 15.0), 3.5)
        self.assertIsNone(module.error_response_time(session, "WRONG_PART", None))

    def test_stage_route_parsing(self):
        self.assertEqual(
            module.parse_stage_route("Stage_01>Stage_02;Stage_03|Stage_04"),
            ["Stage_01", "Stage_02", "Stage_03", "Stage_04"],
        )

    def test_completed_session_derives_route_from_transitions(self):
        # The live ProcessLogger writes a session_end record, but the summary
        # does not necessarily include stage_route/stages_completed. The parser
        # must still derive them from the transition records.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "completed.jsonl"
            records = [
                {"type": "session_start", "session_id": "completed"},
            ]
            previous = None
            for i in range(1, 8):
                stage = f"Stage_{i:02d}"
                records.append({
                    "type": "stage_transition",
                    "elapsed_s": float(i * 10),
                    "from_stage": previous,
                    "to_stage": stage,
                })
                previous = stage
            records.append({
                "type": "session_end",
                "total_time_s": 80.0,
                "states_observed": 19,
                "goal_observed": False,
                "final_product_verified": False,
            })
            p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            parsed = module.parse_session(p)
            self.assertFalse(parsed["reconstructed"])
            self.assertEqual(parsed["summary"]["stage_route"], [f"Stage_{i:02d}" for i in range(1, 8)])
            self.assertEqual(parsed["summary"]["stages_completed"], 7)
            self.assertTrue(parsed["summary"]["goal_reached"])
            self.assertFalse(parsed["summary"]["goal_combination_observed"])
            self.assertFalse(parsed["summary"]["final_product_verified"])


    def test_incomplete_sessions_are_not_in_aggregate_metrics(self):
        base = {
            "condition": "nominal", "log_complete": True,
            "stage_count_agrees": True, "sequence_correct": True,
            "stages_true": 7, "trial_id": "T2-01", "build_time_s": 100.0,
            "goal_reached": True, "warnings_total": 0,
            "over_reported": 0, "under_reported": 0,
        }
        incomplete = dict(base, log_complete=False, trial_id="T2-02", build_time_s=None)
        result = module.aggregate([base, incomplete], [])
        self.assertEqual(result["trials_nominal"], 2)  # recorded
        self.assertEqual(result["trials_nominal_complete"], 1)  # scored
        self.assertEqual(result["stage_count_checked"], 1)
        self.assertEqual(result["build_time_mean_s"], 100.0)

    def test_template_does_not_prefill_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            template = module.write_trial_template(folder, [{"file": "run.jsonl"}])
            rows = module.read_trials(template)
            row = rows["run.jsonl"]
            self.assertEqual(row["condition"], "")
            self.assertEqual(row["true_stages_completed"], "")
            self.assertEqual(row["error_elapsed_s"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
