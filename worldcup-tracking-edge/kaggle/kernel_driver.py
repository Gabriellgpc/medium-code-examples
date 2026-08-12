"""Two experiments, one per GPU, concurrently.

DataParallel measured 0.81x against a single GPU — the model is small and the
heatmap outputs are large, so gathering them dominates. In a phase that compares
configurations, two concurrent runs beat one faster run anyway: two answers
instead of one answer sooner. So each arm gets its own device via
CUDA_VISIBLE_DEVICES and they run side by side.

**Batch 8 and lr 1e-3 in both arms**, identical to every prior run. Batch 16
measured 16% faster, but adopting it here would change the optimisation regime in
the same run that changes capacity and output stride, and neither result would be
attributable. The concurrency buys the wall-clock instead, at no cost to
comparability.

  GPU 0 — w32, 6 epochs.  Capacity won the last comparison on exactly the axis
          twelve epochs could not move (AP75 +13.7%, small objects +16.6%) and was
          free to train. Does it compound with more epochs?

  GPU 1 — w18 with kp_stride 2, 4 epochs, directly comparable to the w18 4-epoch
          run that produced the current pitch numbers. The pitch head now finds
          81% of landmarks but places them at a median 8.78 native px against a
          2-3 px budget, and at stride 4 one output pixel spans 12 native ones.
          The decoding floor was measured at 0.66 px with a clean Gaussian; a
          noisy prediction has far fewer pixels to fit. Halving the stride is the
          cheap test of whether that is the limit.
"""

import json
import os
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
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
sh("nproc", check=False)
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

COMMON = [
    "--heads", "ball,detection,pitch", "--batch", "8", "--lr", "1e-3",
    "--val-stride", "29", "--val-limit", "1500",
    # 4 vCPUs shared by two concurrent trainings; 4 workers each would thrash.
    "--workers", "2", "--eval-every", "2",
]
ARMS = [
    ("cap_w32", "0", ["--trunk-width", "32", "--epochs", "6"]),
    ("pitch_s2", "1", ["--trunk-width", "18", "--epochs", "4",
                       "--kp-stride", "2", "--pitch-upsample", "2"]),
]

procs = []
for tag, gpu, extra in ARMS:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    log = open(OUT / f"{tag}.log", "w")
    cmd = [sys.executable, str(scripts_dir / "train_snet.py"), *COMMON, "--tag", tag, *extra]
    print(f"launching {tag} on GPU {gpu}: {' '.join(cmd[-6:])}", flush=True)
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

t0 = time.time()
for tag, proc, log in procs:
    rc = proc.wait()
    log.close()
    print(f"{tag} exited rc={rc} after {time.time() - t0:.0f}s", flush=True)

summary = {}
for tag, _, _ in procs:
    hist = OUT / "snet" / tag / "history.json"
    entry = {}
    if hist.exists():
        blob = json.loads(hist.read_text())
        h = blob["history"]
        entry = {
            "params_m": blob.get("params_m"),
            "epoch_seconds": [e["seconds"] for e in h],
            "samples_per_s": [e.get("samples_per_s") for e in h],
            "peak_vram_gb": [e.get("peak_vram_gb") for e in h],
            "ball_f1": [e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e],
            "pitch": [
                {k: e["pitch_val"].get(k)
                 for k in ("median_px", "p90_px", "pct_under_budget", "detected_fraction")}
                for e in h if "pitch_val" in e
            ],
            "kendall": [e.get("weights") for e in h][-1:],
        }
    ck = OUT / "snet" / tag / "best.pt"
    if ck.exists():
        subprocess.run([
            sys.executable, str(scripts_dir / "eval_step3.py"),
            "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
            "--val-stride", "87", "--val-limit", "500", "--skip-rfdetr",
            "--snet-ckpt", f"{tag}={ck}", "--out", str(OUT / f"map_{tag}.json"),
        ], check=False)
        f = OUT / f"map_{tag}.json"
        if f.exists():
            entry["map"] = json.loads(f.read_text())["results"].get(tag)
    summary[tag] = entry

summary["_reference_w18_4ep_b8"] = {
    "mAP": 0.2740, "ball_f1": 0.4913,
    "pitch": {"median_px": 8.781, "detected_fraction": 0.8107, "pct_under_budget": 11.48},
}
summary["_reference_w32_4ep_b8"] = {"mAP": 0.2865, "ball_f1": 0.5104}
(OUT / "concurrent_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:4000], flush=True)
print("\ndone", flush=True)
