"""Temporal smoothing of the homography, on contiguous frames."""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru", "albumentations")


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker}")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("eval_temporal.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["valid"]
t0 = time.time()
root = prepare_gsr.fetch()
if not (prepare_gsr.OUT / "valid" / "detection.json").exists():
    prepare_gsr.prepare(root / "valid", "valid")
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

hits = sorted(Path("/kaggle/input").rglob("cap_w32/best.pt"))
if not hits:
    raise SystemExit("checkpoint not mounted")
print(f"checkpoint: {hits[0]}", flush=True)

subprocess.run([
    sys.executable, str(scripts_dir / "eval_temporal.py"),
    "--ckpt", str(hits[0]),
    "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
    "--sequences", "4", "--windows", "3,5,9,15",
    "--out", str(OUT / "temporal.json"),
], check=False)

if (OUT / "temporal.json").exists():
    print(json.dumps(json.loads((OUT / "temporal.json").read_text()), indent=2)[:2500],
          flush=True)
print("\ndone", flush=True)
