"""Four-minute RAM diagnostic: is nn.DataParallel the leak?

Section 9.16 left the leak undiagnosed after a synthetic reproduction came back
clean. The two candidates it named are separable by a single experiment, and this
is that experiment rather than another five-hour run that dies at the same step.

Round 1 tested the allocator (default versus MALLOC_ARENA_MAX=2) and found both
flat: 2.611 GB from step 25 to step 400, with malloc_trim reclaiming nothing
because there was nothing to reclaim. That result only mattered once the reason
was found — the diagnostic ran on one GPU, while `train_snet.py` wraps the model
in `nn.DataParallel` whenever Kaggle hands out two T4s, which it does for the
NvidiaTeslaT4 shape. The leaking run had DataParallel; the flat diagnostic did not.

So round 2 is that one variable, with its own control in the same session:

  A_plain          — one GPU, no wrapper. Expected flat, and reproduces round 1.
  B_dataparallel   — nn.DataParallel across both T4s, exactly as training runs.

If B climbs at roughly 1.64 MB/step and A does not, DataParallel is the cause, and
section 9.9 has already measured it as a throughput *loss* (0.81x) — so the fix is
to stop using it, which costs nothing and gains the run back.

Only the train split is prepared: 400 steps never touch validation, and fetching
valid as well would cost more wall-clock than the measurement.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
STEPS = 400


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
sh("bash", "-lc", "ldd --version | head -1", check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru", "albumentations")


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker}")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("train_snet.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["train"]
t0 = time.time()
root = prepare_gsr.fetch()
if not (prepare_gsr.OUT / "train" / "detection.json").exists():
    prepare_gsr.prepare(root / "train", "train")
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

report = {"steps": STEPS, "passes": []}
for label, dp in (("A_plain", False), ("B_dataparallel", True)):
    env = dict(os.environ)
    env.pop("MALLOC_ARENA_MAX", None)
    out = OUT / f"diag_{label}.json"
    print(f"\n{'=' * 60}\n=== pass {label}\n{'=' * 60}", flush=True)
    t0 = time.time()
    subprocess.run([
        sys.executable, str(scripts_dir / "diag_rss.py"),
        "--steps", str(STEPS), "--batch", "8", "--workers", "2",
        "--trunk-width", "32", "--landmark-set", "expanded",
        "--label", label, "--out", str(out),
    ] + (["--data-parallel"] if dp else []), env=env, check=False)
    print(f"pass {label} took {time.time() - t0:.0f}s", flush=True)
    if out.exists():
        report["passes"].append(json.loads(out.read_text()))

# The comparison the whole kernel exists to make, on one line.
summary = [
    {k: p[k] for k in ("label", "data_parallel", "n_gpu", "torch",
                       "slope_measured_from_step",
                       "parent_growth_mb_per_step", "children_growth_mb_per_step",
                       "trim_reclaimed_at_end_gb")}
    for p in report["passes"]
]
report["summary"] = summary
(OUT / "diag_summary.json").write_text(json.dumps(report, indent=2))
print("\n=== SUMMARY (real run grew 1.64 MB/step) ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("\ndone", flush=True)
