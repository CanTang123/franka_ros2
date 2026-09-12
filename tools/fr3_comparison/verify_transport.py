#!/usr/bin/env python3
"""Start our own loopback service, send all four methods, and tear it down."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from protocol import SCHEMA, JOINTS, response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--frozen-manifest", required=True)
    p.add_argument("--extension", required=True)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=18765)
    args = p.parse_args()
    trial = json.loads(args.trial.read_text())
    req = dict(schema=SCHEMA, method="A", joint_names=JOINTS, request_id=1,
        source=trial["source"], control_step=0, condition_id=trial["condition_id"],
        descriptor=trial["path_descriptor"], state=trial["initial_q_rad"] + trial["initial_dq_rad_s"], phase_time_s=0.)
    def post(value):
        with urlopen(Request(f"http://127.0.0.1:{args.port}/plan", data=json.dumps(value).encode(),
                             headers={"Content-Type": "application/json"}), timeout=1.) as handle:
            return json.load(handle)
    cmd = [sys.executable, str(Path(__file__).with_name("inference_server.py")),
           "--project", args.project, "--frozen-manifest", args.frozen_manifest,
           "--extension", args.extension, "--trial", str(args.trial), "--port", str(args.port)]
    with args.output.with_suffix(".server.log").open("x") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Inference server exited; see server log")
                try:
                    post(req)
                    break
                except URLError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.1)
            rows = []
            for i, method in enumerate("ABCD"):
                value = dict(req, method=method, request_id=i + 2)
                start = time.perf_counter()
                result = post(value)
                response(result, value)
                rows.append(dict(method=method, loopback_roundtrip_s=time.perf_counter() - start,
                                 inference_s=result["inference_s"]))
            try:
                post(dict(req, condition_id="wrong-trial"))
            except HTTPError as error:
                if error.code != 400:
                    raise
            else:
                raise AssertionError("Server accepted the wrong trial")
            report = dict(passed=True, rows=rows, wrong_trial_rejected=True,
                          scope="CPU loopback HTTP; no SSH/WAN/ROS/hardware verification")
            with args.output.open("x") as file:
                json.dump(report, file, indent=2)
            print(json.dumps(report))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
