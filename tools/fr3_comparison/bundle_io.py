"""Resolve portable payload paths without changing recorded training identities."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Bundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "bundle.json").read_text())
        if self.manifest.get("schema") != "fr3_local_inference_bundle_v1":
            raise ValueError("Unsupported local inference bundle")
        for name, expected in self.manifest["files"].items():
            if sha(self.path(name)) != expected:
                raise ValueError(f"Bundle file missing or changed: {name}")

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Bundle path escapes its root: {relative}")
        return path

    def original(self, historical):
        return self.path(self.manifest["original_paths"][str(historical)])
