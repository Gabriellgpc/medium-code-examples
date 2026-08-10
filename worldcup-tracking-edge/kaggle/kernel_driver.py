"""Steps 1 and 2 in one session: does sharing a backbone hurt?

Three runs at an identical budget — ball alone, detection alone, then both on one
trunk. Equal epochs on the full train split, equal validation, so the comparison
is like-for-like. That equality is the whole point: a joint model's number means
nothing without the solo number beside it.

**Why the WASB-faithful trunk is not in this run.** The previous session measured
it at 5.4x the training time per epoch (930 s against 153 s for the same 500
steps). Extrapolated to convergence on the full split that is roughly 36 GPU-hours
against 7, which is more than the entire free weekly quota. It is not just too
slow to *ship* at 214 ms on the iGPU — it is too slow to *train* here. Dropping it
is a budget fact, not a claim that it would lose.

**Why the pitch head is not in this run.** Its target is 33 channels at 384x640,
which is 32 MB per sample and 260 MB per batch of 8 before any augmentation. That
would bottleneck the loader and distort the timing comparison this run exists to
make. The fix is to emit the keypoint target at a coarser stride and rely on
soft-argmax for sub-pixel recovery; until that is written and checked, pitch waits.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

SRC = Path("/kaggle/input/soccernet-tracking-edge-src")
SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")

EPOCHS = 3
BATCH = 8
VAL_STRIDE = 29          # ~1500 val frames spread over all 58 valid sequences
RUNS = [
    ("solo-ball", ["--heads", "ball", "--tag", "solo_ball"]),
    ("solo-detection", ["--heads", "detection", "--tag", "solo_detection"]),
    ("joint-ball-detection", ["--heads", "ball,detection", "--tag", "joint_ball_det"]),
]


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
print("=== environment", flush=True)
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
sh("df", "-h", "/kaggle/working", str(SCRATCH), check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru")


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker} under /kaggle/input")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("train_snet.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))
print(f"package: {pkg_dir}\nscripts: {scripts_dir}", flush=True)

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["train", "valid"]
print("\n=== fetch + prepare", flush=True)
t0 = time.time()
root = prepare_gsr.fetch()
for split in prepare_gsr.SPLITS:
    if not (prepare_gsr.OUT / split / "detection.json").exists():
        prepare_gsr.prepare(root / split, split)
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

summary = {}
for label, extra in RUNS:
    print(f"\n{'=' * 60}\n=== {label}\n{'=' * 60}", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, str(scripts_dir / "train_snet.py"),
        "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--val-stride", str(VAL_STRIDE), "--val-limit", "1500",
        "--workers", "2", *extra,
    ]
    try:
        subprocess.run([str(c) for c in cmd], check=True)
    except subprocess.CalledProcessError:
        traceback.print_exc()
    summary[label] = {"seconds": round(time.time() - t0, 1)}

print("\n=== results", flush=True)
for hist in sorted((OUT / "snet").rglob("history.json")):
    data = json.loads(hist.read_text())
    name = hist.parent.name
    row = {
        "params_m": data.get("params_m"),
        "epoch_seconds": [e["seconds"] for e in data["history"]],
        "final_val_loss": data["history"][-1].get("val_loss"),
    }
    balls = [e["ball_val"]["4"] for e in data["history"] if "ball_val" in e]
    if balls:
        row["ball_f1_at_4px_per_epoch"] = [b["f1"] for b in balls]
        row["best_ball_f1_at_4px"] = max(b["f1"] for b in balls)
        row["best_ball_recall"] = max(b["recall"] for b in balls)
    summary.setdefault(name, {}).update(row)
    print(f"  {name}: {json.dumps(row)}", flush=True)

(OUT / "step12_summary.json").write_text(json.dumps({
    "note": "equal-budget solo vs joint; 3 epochs on the full train split, "
            "not trained to convergence",
    "epochs": EPOCHS, "batch": BATCH, "val_stride": VAL_STRIDE,
    "runs": summary,
}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
print("\ndone", flush=True)
