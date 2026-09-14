import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_replay", ROOT / "tools/fr3_comparison/fixed_replay.py")
fixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixed)
REPORT_SPEC = importlib.util.spec_from_file_location(
    "fixed_report_results", ROOT / "tools/fr3_comparison/report_results.py")
report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(report)


class FixedReplayTests(unittest.TestCase):
    def test_all_slow_sequences_have_exact_transactional_coverage(self):
        for method in "ABCD":
            sequence = fixed.load_sequence(ROOT / "fr3_joint_sequences", method)
            self.assertTrue(sequence.debug_slow5x)
            self.assertEqual(sequence.dt_s, .05)
            cursor = fixed.ReplayCursor(sequence)
            indices = []
            while not cursor.done:
                block = cursor.reserve()
                with self.assertRaisesRegex(ValueError, "already pending"):
                    cursor.reserve()
                reply = fixed.simulated_reply(block, 50, sequence)
                self.assertEqual(reply["target_q_rad"][:4], block["targets"])
                indices.extend(range(block["start_index"], block["end_index"] + 1))
                cursor.commit(block["request_id"])
            self.assertEqual(indices, list(range(1, len(sequence.points))))

    def test_invalid_delay_and_stale_commit_are_rejected(self):
        sequence = fixed.load_sequence(ROOT / "fr3_joint_sequences", "A")
        cursor = fixed.ReplayCursor(sequence)
        block = cursor.reserve()
        with self.assertRaises(ValueError):
            fixed.simulated_reply(block, 20, sequence)
        with self.assertRaisesRegex(ValueError, "stale"):
            cursor.commit(block["request_id"] + 1)

    def test_execution_requires_three_described_confirmations(self):
        data = json.loads((ROOT / "tools/fr3_comparison/site_config.example.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "site.json"
            path.write_text(json.dumps(data))
            fixed.load_site_config(path, require_confirmed=False)
            with self.assertRaisesRegex(ValueError, "unconfirmed"):
                fixed.load_site_config(path, require_confirmed=True)

    def test_current_moveit_acceleration_limits_block_all_sequences(self):
        limits = [3.75, 1.875, 2.5, 3.125, 3.75, 5.0, 5.0]
        for method in "ABCD":
            metrics = fixed.motion_metrics(fixed.load_sequence(ROOT / "fr3_joint_sequences", method), limits)
            self.assertTrue(metrics["acceleration_violations"])

    def test_generator_produces_stationary_safe_blocks_that_pass_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "safe"
            subprocess.run([sys.executable,
                str(ROOT / "tools/fr3_comparison/generate_safe_sequences.py"),
                "--source", str(ROOT / "fr3_joint_sequences"),
                "--output", str(output),
                "--site-config", str(ROOT / "tools/fr3_comparison/site_config.example.json")],
                check=True, capture_output=True, text=True)
            site = fixed.load_site_config(
                ROOT / "tools/fr3_comparison/site_config.example.json")
            for method in "ABCD":
                sequence = fixed.load_sequence(output, method)
                self.assertTrue(sequence.safe_smooth)
                self.assertFalse(sequence.debug_slow5x)
                self.assertEqual(sequence.duration_s, 132.)
                for index in range(0, len(sequence.points), 4):
                    self.assertEqual(sequence.velocities[index], (0.,) * 7)
                    self.assertEqual(sequence.accelerations[index], (0.,) * 7)
                metrics = fixed.motion_metrics(sequence, site)
                for key in ("position_violations", "step_violations", "speed_violations",
                            "acceleration_violations"):
                    self.assertFalse(metrics[key])
                cursor = fixed.ReplayCursor(sequence)
                block = cursor.reserve()
                reply = fixed.simulated_reply(block, 100, sequence)
                self.assertEqual(len(reply["target_dq_rad_s"]), 16)
                self.assertEqual(len(reply["target_ddq_rad_s2"]), 16)

    def test_continuous_generator_stays_close_and_has_safe_fallback_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "continuous"
            subprocess.run([sys.executable,
                str(ROOT / "tools/fr3_comparison/generate_continuous_sequences.py"),
                "--source", str(ROOT / "fr3_joint_sequences"),
                "--output", str(output),
                "--site-config", str(ROOT / "tools/fr3_comparison/site_config.example.json")],
                check=True, capture_output=True, text=True)
            site = fixed.load_site_config(
                ROOT / "tools/fr3_comparison/site_config.example.json")
            for method in "ABCD":
                sequence = fixed.load_sequence(output, method)
                self.assertTrue(sequence.continuous_smooth)
                self.assertEqual(sequence.duration_s, 44.)
                metadata = json.loads(sequence.path.read_text())
                self.assertLessEqual(metadata["maximum_source_deviation_rad"], .0035)
                metrics = fixed.motion_metrics(sequence, site)
                for key in ("position_violations", "step_violations", "speed_violations",
                            "acceleration_violations"):
                    self.assertFalse(metrics[key])
                for index in range(4, len(sequence.points)-1, 4):
                    tail = fixed.braking_tail(sequence.points[index],
                        sequence.velocities[index], sequence.accelerations[index], site)
                    self.assertEqual(tail["terminal_velocity"], [0.] * 7)
                    self.assertEqual(tail["terminal_acceleration"], [0.] * 7)

    def test_report_exports_fixed_indices_and_delay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / "run.jsonl"
            rows = [
                dict(event="initialized", monotonic_s=1., run_id=0, method="fixed-A",
                     experiment_type="simulated_delay_fixed_replay", model_called=False,
                     simulated_delay_ms=50, dry_run=True, trial={}, site_config=None),
                dict(event="enabled", monotonic_s=2., run_id=1),
                dict(event="measured_state", monotonic_s=2., sample_monotonic_s=2., run_id=1,
                     state=[0.]*14, stamp_ns=1, ros_time_ns=1),
                dict(event="plan_ready", monotonic_s=2.07, run_id=1,
                     request={"request_id": 1}, trajectory_start_index=1, trajectory_end_index=4,
                     simulated_delay_ms=50, request_to_send_s=.07,
                     stage_timings_s={"inference_reply_s": .05, "collision_checks_s": .02}),
                dict(event="controller_feedback", monotonic_s=2.08, run_id=1, sequence=1,
                     desired_q_rad=[.01]*7, actual_q_rad=[.009]*7, error_q_rad=[.001]*7),
                dict(event="stopped", monotonic_s=2.1, run_id=1,
                     reason="fixed trajectory dry-run complete"),
                dict(event="recording_complete", monotonic_s=2.2, run_id=1),
            ]
            log.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = report.export_report(log, root / "report", plots=False)
            self.assertEqual(summary["experiment_type"], "simulated_delay_fixed_replay")
            timing = (root / "report/delay_timing.csv").read_text()
            self.assertIn("1,1,4,50,70.0,50.0,20.0", timing)


if __name__ == "__main__":
    unittest.main()
