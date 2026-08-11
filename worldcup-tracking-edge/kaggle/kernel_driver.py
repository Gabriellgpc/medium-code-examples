"""Validate the pitch fix, and pull the one lever the converged run pointed at.

Two arms at an identical budget, differing only in trunk width.

The w18 arm doubles as the check that the pitch head now learns. The 12-epoch run
left it firing about two landmarks per frame out of nine, because its loss was
plain MSE over a target that is 99.7% zeros — predicting nothing scored 2.3e-04
against 0.0 for predicting the truth, and Kendall weighting responded by driving
that task's weight to 55,652. The loss is now the focal form the other two heads
use, which scores "predict nothing" at 0.2954 instead. `evaluate_pitch` reports
landmark error in native pixels and the fraction of landmarks actually found, so
this time the answer does not depend on reading a loss value.

The w32 arm is the lever. Twelve epochs bought +9.4% AP50 and -1.8% AP75: the
model learned to find objects and not to place them, which is an output-stride and
capacity limit rather than a training-budget one. Width is the cheaper of the two
remaining knobs and now fits the latency target.

Four epochs each — enough to separate two architectures, and deliberately not a
convergence run, since convergence was just measured to be worth very little.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
EPOCHS = 4
BATCH = 8
RUNS = [
    ("w18", ["--trunk-width", "18", "--tag", "fixed_w18"]),
    ("w32", ["--trunk-width", "32", "--tag", "fixed_w32"]),
]


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

summary = {}
for label, extra in RUNS:
    print(f"\n{'=' * 60}\n=== {label}\n{'=' * 60}", flush=True)
    t0 = time.time()
    try:
        subprocess.run([
            sys.executable, str(scripts_dir / "train_snet.py"),
            "--heads", "ball,detection,pitch",
            "--epochs", str(EPOCHS), "--batch", str(BATCH),
            "--val-stride", "29", "--val-limit", "1500",
            "--workers", "4", "--eval-every", "2", *extra,
        ], check=True)
    except subprocess.CalledProcessError:
        traceback.print_exc()
    summary[label] = {"seconds": round(time.time() - t0, 1)}

# Score both on the same 500 frames Step 3 used, so the numbers slot straight in.
for label, tag in (("w18", "fixed_w18"), ("w32", "fixed_w32")):
    ck = OUT / "snet" / tag / "best.pt"
    if not ck.exists():
        continue
    subprocess.run([
        sys.executable, str(scripts_dir / "eval_step3.py"),
        "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
        "--val-stride", "87", "--val-limit", "500", "--skip-rfdetr",
        "--snet-ckpt", f"{tag}={ck}", "--out", str(OUT / f"map_{tag}.json"),
    ], check=False)
    f = OUT / f"map_{tag}.json"
    if f.exists():
        summary[label]["map"] = json.loads(f.read_text())["results"].get(tag)
    hist = OUT / "snet" / tag / "history.json"
    if hist.exists():
        h = json.loads(hist.read_text())["history"]
        summary[label]["params_m"] = json.loads(hist.read_text()).get("params_m")
        summary[label]["ball_f1"] = [e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e]
        summary[label]["pitch"] = [e.get("pitch_val") for e in h if "pitch_val" in e]
        summary[label]["epoch_seconds"] = [e["seconds"] for e in h]
        summary[label]["kendall"] = [e.get("weights") for e in h][-1:]

(OUT / "capacity_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:4000], flush=True)
print("\ndone", flush=True)
