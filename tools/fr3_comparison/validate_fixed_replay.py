#!/usr/bin/env python3
"""Offline validation for fixed-trajectory simulated-delay experiments."""
import argparse
import json
import sys
import time
from pathlib import Path

from fixed_replay import DELAYS_MS, ReplayCursor, load_sequence, load_site_config, motion_metrics, simulated_reply


def validate(root, site_config, methods="ABCD"):
    site = load_site_config(site_config, require_confirmed=False)
    result = {"schema": "fr3_fixed_replay_offline_validation_v1", "experiment_type": "simulated_delay_fixed_replay",
              "model_called": False, "site_config": str(Path(site_config).resolve()), "methods": {}}
    for method in methods:
        sequence = load_sequence(root, method)
        metrics = motion_metrics(sequence, site)
        cursor = ReplayCursor(sequence)
        expected = 1
        requests = 0
        while not cursor.done:
            block = cursor.reserve()
            if block["start_index"] != expected or block["end_index"] != expected + 3:
                raise ValueError("replay cursor duplicated or skipped an index")
            for delay in DELAYS_MS:
                reply = simulated_reply(block, delay, sequence)
                if reply["target_q_rad"][:4] != block["targets"]:
                    raise ValueError("reply changed fixed targets")
            cursor.commit(block["request_id"])
            expected += 4
            requests += 1
        result["methods"][method] = dict(path=str(sequence.path), debug_slow5x=sequence.debug_slow5x,
            safe_smooth=sequence.safe_smooth,
            continuous_smooth=sequence.continuous_smooth,
            points=len(sequence.points), commands=sequence.command_count, requests=requests,
            request_period_s=.2, targets_per_request=4, point_dt_s=.05,
            duration_s=sequence.duration_s, metrics=metrics,
            pass_position_limits=not metrics["position_violations"],
            pass_step_limit=not metrics["step_violations"], pass_speed_limit=not metrics["speed_violations"],
            pass_acceleration_limits=not metrics["acceleration_violations"])
    result["passed"] = all(all(row[key] for key in ("pass_position_limits", "pass_step_limit",
        "pass_speed_limit", "pass_acceleration_limits")) for row in result["methods"].values())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--site-config", type=Path, required=True)
    p.add_argument("--methods", default="ABCD")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not args.methods or any(x not in "ABCD" for x in args.methods) or len(set(args.methods)) != len(args.methods):
        p.error("--methods must contain unique labels from ABCD")
    result = validate(args.sequences, args.site_config, args.methods)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
