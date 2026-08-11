"""Batch 16: does the second T4 pay, and does the bigger batch cost accuracy?

Batch has been 8 since Step 1 for comparability, not because of a memory limit,
and every run so far used one of the two T4s Kaggle actually allocates. Two things
change at once if you simply switch to "batch 16 on both GPUs", so this separates
them into two arms:

  A  batch 16, ONE GPU   -> isolates the batch size
  B  batch 16, TWO GPUs  -> isolates the second device

Throughput and accuracy are reported separately on purpose. A larger batch with an
unchanged learning rate and the same OneCycle schedule is a different optimisation
regime, so "faster" and "better" are not the same claim and a drop in F1 would be
the schedule, not the hardware.

Baseline to compare against, same code and config at batch 8 on one GPU:
1,861 s/epoch, ball F1@4px 0.4913 after 4 epochs, mAP 0.2740.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
EPOCHS = 3
BATCH = 16


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
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

prepare_gsr.SPLITS = ["train", "valid"]
t0 = time.time()
root = prepare_gsr.fetch()
for split in prepare_gsr.SPLITS:
    if not (prepare_gsr.OUT / split / "detection.json").exists():
        prepare_gsr.prepare(root / split, split)
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

RUNS = [
    ("b16_1gpu", {"CUDA_VISIBLE_DEVICES": "0"}),
    ("b16_2gpu", {}),
]
summary = {}
for tag, extra_env in RUNS:
    print(f"\n{'=' * 60}\n=== {tag}\n{'=' * 60}", flush=True)
    env = {**os.environ, **extra_env}
    t0 = time.time()
    try:
        subprocess.run([
            sys.executable, str(scripts_dir / "train_snet.py"),
            "--heads", "ball,detection,pitch",
            "--epochs", str(EPOCHS), "--batch", str(BATCH),
            "--trunk-width", "18",
            "--val-stride", "29", "--val-limit", "1500",
            "--workers", "4", "--eval-every", "1", "--tag", tag,
        ], check=True, env=env)
    except subprocess.CalledProcessError:
        traceback.print_exc()
    summary[tag] = {"wall_seconds": round(time.time() - t0, 1)}
    hist = OUT / "snet" / tag / "history.json"
    if hist.exists():
        h = json.loads(hist.read_text())["history"]
        summary[tag]["epoch_seconds"] = [e["seconds"] for e in h]
        summary[tag]["samples_per_s"] = [e.get("samples_per_s") for e in h]
        summary[tag]["peak_vram_gb"] = [e.get("peak_vram_gb") for e in h]
        summary[tag]["n_gpu"] = h[-1].get("n_gpu")
        summary[tag]["ball_f1"] = [e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e]
        summary[tag]["pitch"] = [
            {k: e["pitch_val"].get(k) for k in ("median_px", "detected_fraction")}
            for e in h if "pitch_val" in e
        ]

summary["_baseline_batch8_1gpu"] = {
    "epoch_seconds": 1861, "ball_f1_after_4_epochs": 0.4913, "mAP": 0.2740,
}
(OUT / "batch_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:3000], flush=True)
print("\ndone", flush=True)
