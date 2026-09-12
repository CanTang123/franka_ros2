"""Frozen FR3 A/B/C/D inference from current, hash-checked mathematical source."""
import hashlib
import json
from pathlib import Path
import sys
import time

from protocol import JOINTS, SCHEMA, request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runtime:
    def __init__(self, project, frozen_manifest, extension):
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
            if sha(historical) != expected:
                raise ValueError(f"Frozen data/model changed: {historical}")
        for historical, expected in identity["extension_source"].items():
            if sha(extension / Path(historical).name) != expected:
                raise ValueError(f"Native extension changed: {historical}")
        sys.path[:0] = [str(project / "src"), str(project.parent / "DynaFlow"), str(extension)]
        import jax
        import jax.numpy as jnp
        import numpy as np
        import mujoco
        from dm_cfm.mean_flow_decoder.adapters import make
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
        cfg = dict(self.frozen["config"], lambda_stop=.5)
        self.adapter = make(cfg, False)
        self.native = NativeAdjoint(self.adapter.native, workers=1)
        self.adapter.rollout = self.native.rollout
        self.mean = Component(cfg, "mean", self.adapter)
        self.decoder = Component(cfg, "mf_det", self.adapter)
        self.checkpoints = {name: self.frozen["checkpoints"][name] for name in ("mean25k", "decoder10k")}
        params = []
        for name, entry in self.checkpoints.items():
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
            "inference_s": time.perf_counter() - started, "checkpoints": self.checkpoints}
