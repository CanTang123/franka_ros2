#!/usr/bin/env python3
"""Import recorded FR3 targets without loading models or connecting to ROS."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = {"A": "full", "B": "dynamics", "C": "direct", "D": "bypass"}
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def export_results(results_root, output, condition_id=None, source=0, all_records=False):
    root, output = Path(results_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    manifests = {m: json.loads((root / v / "RESULTS.json").read_text()) for m, v in METHODS.items()}
    if condition_id is None and not all_records:
        condition_id = manifests["A"]["rows"][0]["condition_id"]
    selection = {}
    for method, manifest in manifests.items():
        rows = [r for r in manifest["rows"] if r["kind"] == "closed_loop" and
                (all_records or (r["condition_id"] == condition_id and r["source"] == source))]
        ids = [(r["condition_id"], r["source"]) for r in rows]
        if not rows or len(ids) != len(set(ids)):
            raise ValueError(f"Missing or duplicate cases for {method}")
        selection[method] = {key: r for key, r in zip(ids, rows)}
    if any(set(rows) != set(selection["A"]) for rows in selection.values()):
        raise ValueError("Methods do not contain the same condition/source pairs")
    # Validate the entire selection before creating any output.
    bundles = []
    for method, rows in selection.items():
        for index, (key, row) in enumerate(sorted(rows.items())):
            record = Path(row["record"])
            if sha(record) != row["sha256"]:
                raise ValueError(f"Source SHA256 mismatch: {record}")
            with np.load(record, allow_pickle=False) as data:
                actions = np.array(data["actions"], dtype=float)
                states = np.array(data["states"], dtype=float)
                times = np.array(data["times"], dtype=float)
                descriptor = np.array(data["descriptor"], dtype=float)
            n = len(actions)
            if n < 1 or actions.shape != (n, 7) or states.shape != (n + 1, 14):
                raise ValueError(f"Invalid state/action dimensions: {record}")
            if times.shape != (n + 1,) or descriptor.shape != (19,):
                raise ValueError(f"Invalid times/descriptor: {record}")
            if not all(np.isfinite(a).all() for a in (actions, states, times, descriptor)):
                raise ValueError(f"Nonfinite record: {record}")
            if abs(times[0]) > 1e-9 or not np.allclose(np.diff(times), .05, rtol=0, atol=1e-7):
                raise ValueError(f"Expected FR3 0.05s control periods: {record}")
            bundle = {
                "schema": "fr3_comparison_record_v1", "method": method,
                "variant": METHODS[method], "condition_id": key[0], "source": key[1],
                "mode": "recorded_simulation_targets", "joint_names": JOINTS,
                "units": {"position": "rad", "velocity": "rad/s", "time": "s"},
                "initial_q_rad": states[0, :7].tolist(), "initial_dq_rad_s": states[0, 7:].tolist(),
                "target_q_rad": actions.tolist(), "target_start_times_s": times[:-1].tolist(),
                "control_dt_s": .05, "duration_s": float(times[-1]),
                "path_descriptor": descriptor.tolist(),
                "simulation_completed": bool(row["completed"]),
                "simulation_success": bool(row["success"]),
                "simulation_constraint_violated": bool(row["constraint_violated"]),
                "provenance": {"record": str(record), "record_sha256": sha(record),
                    "results_manifest": str(root / METHODS[method] / "RESULTS.json"),
                    "results_sha256": sha(root / METHODS[method] / "RESULTS.json"),
                    "candidate_sha": manifests[method].get("candidate_sha"),
                    "method_definition": manifests[method]["method_definition"]},
                "execution_note": "Recorded simulation targets; replay is not hardware-feedback replanning. "
                    "ROS trajectory interpolation changes the original zero-order-hold servo protocol.",
            }
            bundles.append((f"{method}_{index:03d}_s{key[1]}.json", bundle))
    output.mkdir(parents=True)
    index = []
    for filename, bundle in bundles:
        write(output / filename, bundle)
        index.append({"file": filename, "sha256": sha(output / filename),
                      "method": bundle["method"], "condition_id": bundle["condition_id"],
                      "source": bundle["source"]})
    write(output / "INDEX.json", {"schema": "fr3_comparison_index_v1", "records": index,
                                 "hardware_connected": False})
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, help="Directory containing full/dynamics/direct/bypass")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--condition-id")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--all", action="store_true", dest="all_records")
    args = parser.parse_args()
    if args.all_records and (args.condition_id is not None or args.source != 0):
        parser.error("--all cannot be combined with a condition/source selection")
    rows = export_results(**vars(args))
    print(json.dumps({"records": len(rows), "output": args.output, "hardware_connected": False}))


if __name__ == "__main__":
    main()
