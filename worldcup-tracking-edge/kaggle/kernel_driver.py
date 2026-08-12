"""Measure the ceiling of the expanded landmark set before paying for a retrain."""

import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
SCRATCH.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "loguru"], check=False)


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker}")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("measure_landmark_ceiling.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

import prepare_gsr  # noqa: E402

# Only the annotations are needed — this is geometry, no frames are read. But
# prepare_gsr builds detection.json from the expanded archive, so the download is
# still the cost.
prepare_gsr.SPLITS = ["valid"]
t0 = time.time()
root = prepare_gsr.fetch()
if not (prepare_gsr.OUT / "valid" / "detection.json").exists():
    prepare_gsr.prepare(root / "valid", "valid")
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

subprocess.run([
    sys.executable, str(scripts_dir / "measure_landmark_ceiling.py"),
    "--gsr", str(OUT / "gsr"), "--split", "valid",
    "--stride", "29", "--limit", "1500",
    "--out", str(OUT / "landmark_ceiling.json"),
], check=False)
print("\ndone", flush=True)
