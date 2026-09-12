#!/usr/bin/env python3
"""Check four frozen first plans against recorded numerical evidence, offline."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_runtime import Runtime
from protocol import SCHEMA, JOINTS, response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = Runtime(args.project, args.frozen_manifest, args.extension)
    index = json.loads((args.records / "INDEX.json").read_text())
    rows = []
    for method in "ABCD":
        entry = next(r for r in index["records"] if r["method"] == method)
        trial = json.loads((args.records / entry["file"]).read_text())
        req = dict(schema=SCHEMA, method=method, joint_names=JOINTS, request_id=1,
            source=trial["source"], control_step=0, condition_id=trial["condition_id"],
            descriptor=trial["path_descriptor"], state=trial["initial_q_rad"] + trial["initial_dq_rad_s"], phase_time_s=0.)
        first = np.array(response(runtime.infer(req), req))
        result = runtime.infer(req)
        actual = np.array(response(result, req))
        with np.load(trial["provenance"]["record"], allow_pickle=False) as evidence:
            expected = evidence["planned_actions"][0]
        error = float(np.max(abs(actual - expected)))
        repeated = float(np.max(abs(actual - first)))
        # One request with changed measured q verifies feedback actually enters
        # the inference path, without creating a new trajectory experiment.
        changed = dict(req, state=list(req["state"]))
        changed["state"][0] += .001
        sensitivity = float(np.max(abs(np.array(runtime.infer(changed)["target_q_rad"]) - actual)))
        rows.append(dict(method=method, max_abs_vs_saved_rad=error, repeat_max_abs_rad=repeated,
                         measured_state_sensitivity_rad=sensitivity, warm_inference_s=result["inference_s"]))
        if error > 2e-6 or repeated != 0 or sensitivity == 0:
            raise AssertionError(rows[-1])
    report = dict(passed=True, rows=rows, checkpoints=runtime.checkpoints,
                  current_source_hashes=runtime.source_hashes,
                  scope="One saved condition/source; full H16; CPU; no ROS or hardware validation")
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
