"""Kaggle driver: fetch SN-GSR-2025 train split, build annotations, run M1-M3.

Orchestration lives here rather than in measure_gsr.main() so that one failing
check cannot cost us the other two: each is wrapped and its result written
independently. A 10 GB download is too expensive to redo because M2 could not
find a bright blob.
"""

import json
import subprocess
import sys
import traceback
from pathlib import Path

SRC = Path("/kaggle/input/soccernet-tracking-edge-src")
SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working/gsr")
MEAS = OUT / "measurements"
SPLIT = "train"
REPO_ID = "SoccerNet/SN-GSR-2025"


def sh(*cmd, check=True):
    """Run a command. Diagnostics pass check=False so they cannot kill the job."""
    print("$", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


# /kaggle/tmp does not exist until something creates it, and `df` on a missing
# path exits non-zero — which is how version 1 of this kernel killed itself on
# its own disk-space diagnostic.
SCRATCH.mkdir(parents=True, exist_ok=True)
MEAS.mkdir(parents=True, exist_ok=True)

print("=== environment", flush=True)
sh("df", "-h", "/kaggle/working", str(SCRATCH), check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru")

print("\n=== /kaggle/input tree", flush=True)
for p in sorted(Path("/kaggle/input").rglob("*"))[:80]:
    print(f"  {'d' if p.is_dir() else 'f'} {p}", flush=True)

# Resolve by glob rather than by assumed layout: how Kaggle expands an uploaded
# archive is not something to hard-code, and v2 of this kernel died guessing.
def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker} under /kaggle/input")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent   # .../src
scripts_dir = locate("measure_gsr.py")
print(f"\npackage root: {pkg_dir}\nscripts:      {scripts_dir}", flush=True)
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

# Pre-flight: confirm the dataset is reachable and see the file sizes before
# committing to a multi-gigabyte download.
print("\n=== hub pre-flight", flush=True)
from huggingface_hub import HfApi  # noqa: E402

info = HfApi().repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
for sib in info.siblings:
    size = sib.size / 1e9 if sib.size else 0.0
    print(f"  {sib.rfilename:<40} {size:6.2f} GB", flush=True)

import measure_gsr  # noqa: E402
import prepare_gsr  # noqa: E402

# Only the train split: each split is a ~10 GB download and M1-M3 need one.
prepare_gsr.SPLITS = [SPLIT]

print("\n=== fetch + prepare", flush=True)
root = prepare_gsr.fetch()
OUT.mkdir(parents=True, exist_ok=True)
summary = prepare_gsr.prepare(root / SPLIT, SPLIT)
sh("du", "-sh", str(root), str(OUT), check=False)

det = OUT / SPLIT / "detection.json"
track = OUT / SPLIT / "ball_track.json"
results = {"split": SPLIT, "prepare_summary": summary}

for name, fn in [
    ("m1_box_geometry", lambda: measure_gsr.m1_box_geometry(det)),
    ("m2_ball_convention", lambda: measure_gsr.m2_ball_convention(track, root / SPLIT)),
    ("m3_keypoint_budget", lambda: measure_gsr.m3_keypoint_budget(det)),
]:
    print(f"\n=== {name}", flush=True)
    try:
        results[name] = fn()
        print(json.dumps(results[name], indent=2)[:4000], flush=True)
    except Exception:
        # Record the failure in the artifact instead of only in the log, so the
        # downloaded result says plainly which check did not run.
        results[name] = {"error": traceback.format_exc()}
        traceback.print_exc()
    (MEAS / f"{SPLIT}_measurements.json").write_text(json.dumps(results, indent=2))

print("\n=== done", flush=True)
sh("ls", "-laR", str(MEAS), check=False)
