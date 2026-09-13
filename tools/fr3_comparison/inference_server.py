#!/usr/bin/env python3
"""Loopback JSON inference service, normally on the same computer as ROS."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path

from model_runtime import Runtime
from protocol import SCHEMA, JOINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Portable local model/data/source payload")
    parser.add_argument("--project")
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--extension")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    runtime = Runtime(args.project, args.frozen_manifest, args.extension, bundle=args.bundle)
    trial = json.loads(args.trial.read_text())
    warmup = dict(schema=SCHEMA, joint_names=JOINTS, request_id=0, source=trial["source"],
                  control_step=0, condition_id=trial["condition_id"], descriptor=trial["path_descriptor"],
                  state=trial["initial_q_rad"] + trial["initial_dq_rad_s"], phase_time_s=0.)
    for method in "ABCD":
        runtime.infer(dict(warmup, method=method))
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status = 200
            try:
                if self.path != "/plan":
                    raise ValueError("Unknown endpoint")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size")
                self.connection.settimeout(1.)
                value = json.loads(self.rfile.read(size))
                # Bind the server to one declared experimental condition.
                for name in ("condition_id", "source"):
                    if value.get(name) != trial[name]:
                        raise ValueError(f"Trial {name} mismatch")
                if value.get("descriptor") != trial["path_descriptor"]:
                    raise ValueError("Trial descriptor mismatch")
                result = runtime.infer(value)
            except Exception as error:
                status, result = 400, {"error": str(error)}
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"ready": True, "bind": f"127.0.0.1:{args.port}", "device_info": runtime.device_info,
                      "methods": list("ABCD"), "checkpoints": runtime.checkpoints,
                      "source_hashes": runtime.source_hashes}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
