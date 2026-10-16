"""Package the built interface for hosts that serve only text and web types.

Copies web/dist to an output folder, base64 encodes every gzip file of the
bundle as <name>.b64.txt, and marks the bundle manifest with
"encoding": "gzip+base64" so the loader decodes before decompressing.
The repository build in web/dist is left untouched.

Run:  python -m allelion.export.artifact <out_dir>
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

from allelion.config import ROOT


def package(out: Path) -> dict:
    dist = ROOT / "web" / "dist"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(dist, out)
    b = out / "bundle"
    for gz in sorted(b.glob("*.gz")):
        (b / (gz.name + ".b64.txt")).write_text(base64.b64encode(gz.read_bytes()).decode())
        gz.unlink()
    m = json.loads((b / "manifest.json").read_text())
    m["encoding"] = "gzip+base64"
    (b / "manifest.json").write_text(json.dumps(m, indent=1))
    files = {str(p.relative_to(out)): str(p.relative_to(out)) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "index.html"}
    size = sum((out / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB in {out}")
    return files


if __name__ == "__main__":
    package(Path(sys.argv[1]))
