"""Frozen FR3 A/B/C/D inference from current, hash-checked mathematical source."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from protocol import JOINTS, SCHEMA, request
from bundle_io import Bundle, sha

# Native MuJoCo uses jax.pure_callback on the host. With JAX 0.6.2,
# CUDA-only initialization excludes the CPU device needed by method B.
# CUDA remains first (the default computation device); failure to initialize
# it must fail rather than silently claim GPU execution on a CPU.
if os.environ.get("JAX_PLATFORMS", "").strip() in ("cuda", "gpu"):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"


class Runtime:
    def __init__(self, project=None, frozen_manifest=None, extension=None, bundle=None):
        payload = Bundle(bundle) if bundle else None
        if payload:
            if any(v is not None for v in (project, frozen_manifest, extension)):
                raise ValueError("Choose --bundle OR the legacy project paths")
            project = payload.path("source/dynamics_manifold_cfm")
            extension = payload.path("source/native_extension")
            frozen_manifest = payload.path("FROZEN_CANDIDATES.json")
        elif any(v is None for v in (project, frozen_manifest, extension)):
            raise ValueError("Provide --bundle for local inference, or all three legacy paths")
        project, extension = Path(project).resolve(), Path(extension).resolve()
        self.frozen = json.loads(Path(frozen_manifest).read_text())
        identity = self.frozen["training_identity"]
        self.source_hashes = {}
        # Verify the current files actually imported for FR3. Go1's adapter is
        # recorded in the training identity but is not imported on this path.
        for historical, expected in identity["source"].items():
            if historical.endswith("DynaFlow/mean_flow_decoder/adapter.py"):
                continue
            if "/dynamics_manifold_cfm/" in historical:
                current = project / historical.split("/dynamics_manifold_cfm/", 1)[1]
            else:
                current = project.parent / "DynaFlow" / historical.split("/DynaFlow/", 1)[1]
            if sha(current) != expected:
                raise ValueError(f"Current mathematical source differs from frozen FR3 source: {current}")
            self.source_hashes[str(current)] = expected
        for historical, expected in identity["data"].items():
            if sha(payload.original(historical) if payload else historical) != expected:
                raise ValueError(f"Frozen data/model changed: {historical}")
        for historical, expected in identity["extension_source"].items():
            if sha(extension / Path(historical).name) != expected:
                raise ValueError(f"Native extension changed: {historical}")
        sys.path[:0] = [str(project / "src"), str(project.parent / "DynaFlow"), str(extension)]
        import jax
        import jax.numpy as jnp
        import numpy as np
        import mujoco
        from dm_cfm.mean_flow_decoder.adapters import FR3Adapter
        from dm_cfm.mean_flow_decoder.method import Component
        from dm_cfm.mean_flow_decoder.io import load
        from dm_cfm.fr3_core5.data import condition
        from native_adjoint import NativeAdjoint
        if jax.__version__ != identity["runtime"]["jax"]:
            raise ValueError("JAX version differs from the frozen runtime")
        if mujoco.__version__ != "3.2.7":
            raise ValueError("Frozen FR3 rollout requires MuJoCo 3.2.7")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        self.jax, self.jnp, self.np, self.condition = jax, jnp, np, condition
        self.device_info = {"backend": jax.default_backend(),
                            "devices": [str(d) for d in jax.devices()],
                            "device_kinds": [d.device_kind for d in jax.devices()],
                            "physics_backend": "native MuJoCo CPU"}
        cfg = dict(self.frozen["config"], lambda_stop=.5)
        stats_path = Path(cfg["data_root"]) / "data/stats.npz"
        model_path = Path(cfg["model_path"])
        if payload:
            stats_path, model_path = payload.original(stats_path), payload.original(model_path)
        # Keep the original mathematical methods, but do not instantiate the
        # training Dataset or load episodes.npz merely to normalize inference.
        self.adapter = FR3Adapter.__new__(FR3Adapter)
        self.adapter.cfg = cfg
        with np.load(stats_path, allow_pickle=False) as data:
            self.adapter.data = SimpleNamespace(stats=dict(data))
        self.adapter.stats = jax.tree.map(jnp.asarray, self.adapter.data.stats)
        self.adapter.mid = self.adapter.stats['joint_mid']
        self.adapter.half = self.adapter.stats['joint_half']
        self.adapter.center = jnp.asarray([0, -.4, 0, -1.8, 0, 1.8, -.7853], jnp.float32)
        self.adapter.scale = jnp.minimum(self.adapter.center - (self.adapter.mid - self.adapter.half),
                                        self.adapter.mid + self.adapter.half - self.adapter.center)
        if payload:
            # Original XML bytes/hash stay unchanged on disk. Only asset lookup
            # is relocated in memory; physical values and meshes are preserved.
            model = ET.fromstring(model_path.read_text())
            model.find("compiler").set("meshdir", str(payload.path("assets")))
            self.adapter.native = mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"))
        else:
            self.adapter.native = mujoco.MjModel.from_xml_path(str(model_path))
        self.native = NativeAdjoint(self.adapter.native, workers=1)
        self.adapter.rollout = self.native.rollout
        self.mean = Component(cfg, "mean", self.adapter)
        self.decoder = Component(cfg, "mf_det", self.adapter)
        self.checkpoints = {name: dict(self.frozen["checkpoints"][name]) for name in ("mean25k", "decoder10k")}
        params = []
        for name, entry in self.checkpoints.items():
            if payload:
                entry["original_path"] = entry["path"]
                entry["path"] = str(payload.original(entry["path"]))
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
            checkpoint = load(entry["path"])
            if checkpoint["identity"] != identity or checkpoint["completed_updates"] != entry["updates"]:
                raise ValueError(f"Checkpoint identity mismatch: {name}")
            params.append(jax.tree.map(jnp.asarray, checkpoint["ema"]))
        self.mp, self.dp = params
        self.functions = {method: self.make_fn(method) for method in "ABCD"}

    def make_fn(self, method):
        jax, jnp = self.jax, self.jnp
        def generate(state, cond, key):
            x = jax.random.normal(key, (1, 16, 14), dtype=jnp.float32)
            zero = jnp.zeros((1, 1), jnp.float32)
            if method == "A":
                endpoint = self.mean.apply(self.mp, x, zero, cond)
                x = x + jnp.float32(.5) * (endpoint - x)
            elif method == "B":
                a = self.adapter.actions(self.decoder.apply(self.dp, x, zero, cond))
                endpoint = self.adapter.norm(self.adapter.rollout(state, a)[:, 1:]).astype(jnp.float32)
                x = x + jnp.float32(.5) * (endpoint - x)
            return self.adapter.actions(self.decoder.apply(
                self.dp, x, jnp.full((1, 1), 0 if method == "C" else .5, jnp.float32), cond))
        return jax.jit(generate)

    def infer(self, value):
        request(value)
        started = time.perf_counter()
        jax, jnp, np = self.jax, self.jnp, self.np
        state = np.asarray([value["state"]], np.float32)
        desc = np.asarray([value["descriptor"]], np.float32)
        raw = self.condition(state, desc, np.array([value["phase_time_s"]]))
        stats = self.adapter.data.stats
        cond = jnp.asarray((raw - stats["condition_mean"]) / stats["condition_scale"], jnp.float32)
        key = jax.random.fold_in(jax.random.PRNGKey(27182),
            int(hashlib.sha256(value["condition_id"].encode()).hexdigest()[:8], 16))
        key = jax.random.fold_in(key, value["source"])
        key = jax.random.fold_in(key, value["control_step"])
        actions = np.asarray(self.functions[value["method"]](jnp.asarray(state), cond, key))[0]
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid model output")
        return {"schema": SCHEMA, **{k: value[k] for k in (
            "request_id", "method", "condition_id", "source", "control_step")},
            "joint_names": JOINTS, "control_dt_s": .05, "target_q_rad": actions.tolist(),
            "inference_s": time.perf_counter() - started, "checkpoints": self.checkpoints,
            "device_info": self.device_info}
