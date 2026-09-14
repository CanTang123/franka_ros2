import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/fr3_comparison" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module("export_results")
preview = module("preview_result")
protocol = module("protocol")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for variant in exporter.METHODS.values():
            folder = self.root / variant
            folder.mkdir()
            record = folder / "record.npz"
            np.savez(record, states=np.zeros((3, 14)), actions=np.ones((2, 7)),
                     times=np.array([0, .05, .1]), descriptor=np.zeros(19))
            row = dict(record=str(record), sha256=exporter.sha(record), kind="closed_loop",
                       condition_id="case", source=0, completed=True, success=False, constraint_violated=False)
            exporter.write(folder / "RESULTS.json", dict(rows=[row], method_definition={}))

    def test_four_methods_preserve_targets_and_time_semantics(self):
        output = self.root / "output"
        rows = exporter.export_results(self.root, output)
        self.assertEqual({r["method"] for r in rows}, set("ABCD"))
        record = json.loads((output / rows[0]["file"]).read_text())
        self.assertEqual(record["target_q_rad"], [[1.] * 7] * 2)
        self.assertEqual(record["target_start_times_s"], [0., .05])
        payload = preview.trajectory_payload(record, 2.)
        self.assertEqual([p["time_from_start_s"] for p in payload["points"]], [0., .1, .2])
        self.assertFalse(record["simulation_success"])
        for mutate in [lambda r: r["joint_names"].reverse(),
                       lambda r: r["target_q_rad"][0].__setitem__(0, float("nan")),
                       lambda r: r["target_start_times_s"].__setitem__(1, 0.),
                       lambda r: r.__setitem__("duration_s", .2)]:
            invalid = copy.deepcopy(record)
            mutate(invalid)
            with self.assertRaises(ValueError):
                preview.trajectory_payload(invalid)

    def test_hash_mismatch_leaves_no_output(self):
        with (self.root / "full/record.npz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            exporter.export_results(self.root, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_unpaired_cases_rejected(self):
        path = self.root / "bypass/RESULTS.json"
        manifest = json.loads(path.read_text())
        manifest["rows"][0]["source"] = 1
        exporter.write(path, manifest)
        with self.assertRaises(ValueError):
            exporter.export_results(self.root, self.root / "output", all_records=True)

    def test_online_identity_and_guard(self):
        req = dict(schema=protocol.SCHEMA, method="A", joint_names=protocol.JOINTS,
                   state=[0.] * 14, descriptor=[0.] * 18 + [8.], phase_time_s=0.,
                   request_id=2, source=0, control_step=0, condition_id="case")
        protocol.request(req)
        reply = dict(req, target_q_rad=[[.01] * 7] * 16, control_dt_s=.05)
        actions = protocol.response(reply, req)
        protocol.guard_prefix(actions[:4], [0.] * 7, [-1.] * 7, [1.] * 7, .03, .6)
        with self.assertRaises(ValueError):
            protocol.response(dict(reply, request_id=1), req)
        with self.assertRaises(ValueError):
            protocol.response(dict(reply, method="B"), req)
        with self.assertRaises(ValueError):
            protocol.guard_prefix([[.1] * 7], [0.] * 7, [-1.] * 7, [1.] * 7, .03, .6)
        with self.assertRaises(ValueError):
            protocol.guard_prefix([[2.] * 7], [0.] * 7, [-1.] * 7, [1.] * 7, 10., 100.)
        with self.assertRaisesRegex(ValueError, "fr3_joint1"):
            protocol.guard_prefix([[.04] * 7], [0.] * 7, [-1.] * 7, [1.] * 7, .03, .6)
        protocol.guard_prefix([[.04] * 7], [0.] * 7, [-1.] * 7, [1.] * 7,
                              .09, .6, control_dt=.15)
        samples = protocol.collision_samples([0.] * 7, [[.03] * 7])
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[-1], [.03] * 7)


if __name__ == "__main__":
    unittest.main()
