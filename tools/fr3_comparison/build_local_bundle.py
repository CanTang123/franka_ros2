#!/usr/bin/env python3
"""Copy the two original checkpoints and all required inference assets, with hashes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from bundle_io import sha


def build(project, frozen_manifest, extension, trials, output):
    project, extension, trials = [Path(p).resolve() for p in (project, extension, trials)]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(Path(frozen_manifest).read_text())
    ident = frozen["training_identity"]
    mapping, hashes = {}, {}
    def copy(source, destination, original=None, expected=None):
        source = Path(source)
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError(f"Input hash mismatch: {source}")
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise ValueError(f"Copy mismatch: {target}")
        hashes[destination] = digest
        if original is not None:
            mapping[str(original)] = destination
    copy(frozen_manifest, "FROZEN_CANDIDATES.json")
    for original, digest in ident["source"].items():
        if original.endswith("DynaFlow/mean_flow_decoder/adapter.py"):
            continue
        if "/dynamics_manifold_cfm/" in original:
            relative = original.split("/dynamics_manifold_cfm/", 1)[1]
            current, destination = project / relative, "source/dynamics_manifold_cfm/" + relative
        else:
            relative = original.split("/DynaFlow/", 1)[1]
            current, destination = project.parent / "DynaFlow" / relative, "source/DynaFlow/" + relative
        copy(current, destination, original, digest)
    # Package initializers and checkpoint IO are also needed at inference time.
    for name in ("src/dm_cfm/mean_flow_decoder/__init__.py",
                 "src/dm_cfm/mean_flow_decoder/io.py"):
        copy(project / name, "source/dynamics_manifold_cfm/" + name)
    # The research package's public facade eagerly imports unrelated PyTorch
    # training APIs. Retain it as provenance, use a minimal package initializer
    # for this JAX-only subset; mathematical modules above stay byte-identical.
    copy(project / "src/dm_cfm/__init__.py", "provenance/dm_cfm_public_init.py")
    namespace = "source/dynamics_manifold_cfm/src/dm_cfm/__init__.py"
    (out / namespace).write_text('"""Portable FR3 inference subset; original mathematical modules are unchanged."""\n')
    hashes[namespace] = sha(out / namespace)
    for original, digest in ident["extension_source"].items():
        copy(extension / Path(original).name, "source/native_extension/" + Path(original).name, original, digest)
    for original, digest in ident["data"].items():
        dest = "data/" + Path(original).name if "/data/" in original else "model/" + Path(original).name
        copy(original, dest, original, digest)
    for name in ("mean25k", "decoder10k"):
        entry = frozen["checkpoints"][name]
        copy(entry["path"], f"checkpoints/{name}.pkl", entry["path"], entry["sha256"])
    model = ET.fromstring(Path(frozen["config"]["model_path"]).read_text())
    meshdir = Path(model.find("compiler").get("meshdir"))
    for mesh in model.findall("asset/mesh"):
        file = Path(mesh.get("file"))
        if file.is_absolute() or len(file.parts) != 1:
            raise ValueError("This packager expects the validated model's flat mesh paths")
        copy(meshdir / file, "assets/" + str(file), str(meshdir / file))
    for license_file in meshdir.parent.glob("*LICENSE*"):
        if license_file.is_file():
            copy(license_file, "assets/" + license_file.name)
    for trial in sorted(trials.glob("*.json")):
        copy(trial, "trials/" + trial.name)
        if trial.name != "INDEX.json":
            data = json.loads(trial.read_text())
            evidence = data["provenance"]
            copy(evidence["record"], f"evidence/{data['method']}.npz", evidence["record"], evidence["record_sha256"])
    packages = ["jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy", "scipy", "ml-dtypes", "mujoco", "chex"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    requirements = "# CPU inference; Python 3.11; keep separate from ROS's system Python.\n" + "".join(
        f"{name}=={version}\n" for name, version in versions.items())
    (out / "requirements-cpu.txt").write_text(requirements)
    hashes["requirements-cpu.txt"] = sha(out / "requirements-cpu.txt")
    manifest = dict(schema="fr3_local_inference_bundle_v1", files=hashes, original_paths=mapping,
        python=f"{sys.version_info.major}.{sys.version_info.minor}", dependencies=versions,
        checkpoints={name: frozen["checkpoints"][name] for name in ("mean25k", "decoder10k")},
        notes=["Original checkpoint bytes and training identity are unchanged.",
               "Historical absolute paths are provenance only; runtime resolves bundle-local paths.",
               "The meshdir in the original XML is relocated only in memory.",
               "No training episodes or external server connection is needed for inference."])
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "frozen-manifest", "extension", "trials", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    manifest = build(**vars(args))
    print(json.dumps(dict(output=args.output, files=len(manifest["files"]), dependencies=manifest["dependencies"])))
