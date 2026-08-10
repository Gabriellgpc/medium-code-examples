"""Step 3: the control, plus the price of a bug found on the way to it.

Two questions in one session, because the 7-minute data fetch is the fixed cost:

1. **What did the detection-target bug cost?** The Steps 1-2 checkpoint was trained
   with the Gaussian drawn at the float centre while size and offset were stored at
   the floored pixel. A ground-truth round-trip through the decoder scored mAP 0.24
   instead of 1.00 because of it — 1/4, the probability that both fractional parts
   land below 0.5. Retraining at an identical budget with the fixed target and
   scoring both checkpoints on the same frames prices the bug exactly.

2. **How does the detection head compare to RF-DETR?** Same frames, same
   categories, same pycocotools. Accuracy only: RF-DETR-Large here runs at 1280 px
   and 1.3 FPS on the target iGPU against SNet's 384x640 at 32.7 FPS.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
EPOCHS = 3
BATCH = 8


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
print("=== environment", flush=True)
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv", check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru")
# RF-DETR is only needed for the control arm; a failure here must not take the run
# down, so eval_step3 catches the ImportError and records it.
sh(sys.executable, "-m", "pip", "install", "-q", "rfdetr", check=False)


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

# The Steps 1-2 checkpoint, mounted from that kernel's output.
old_ckpts = sorted(Path("/kaggle/input").rglob("joint_ball_det/best.pt"))
old_ckpt = old_ckpts[0] if old_ckpts else None
print(f"previous checkpoint: {old_ckpt}", flush=True)

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["train", "valid"]
print("\n=== fetch + prepare", flush=True)
t0 = time.time()
root = prepare_gsr.fetch()
for split in prepare_gsr.SPLITS:
    if not (prepare_gsr.OUT / split / "detection.json").exists():
        prepare_gsr.prepare(root / split, split)
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

print(f"\n{'=' * 60}\n=== retrain joint with the fixed detection target\n{'=' * 60}", flush=True)
t0 = time.time()
try:
    subprocess.run([
        sys.executable, str(scripts_dir / "train_snet.py"),
        "--heads", "ball,detection", "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--val-stride", "29", "--val-limit", "1500", "--workers", "2",
        "--tag", "joint_fixed_target",
    ], check=True)
except subprocess.CalledProcessError:
    traceback.print_exc()
train_seconds = round(time.time() - t0, 1)

print(f"\n{'=' * 60}\n=== Step 3: mAP\n{'=' * 60}", flush=True)
new_ckpt = OUT / "snet" / "joint_fixed_target" / "best.pt"
cmd = [
    sys.executable, str(scripts_dir / "eval_step3.py"),
    "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
    "--val-stride", "87", "--val-limit", "500",
    "--out", str(OUT / "step3_map.json"),
]
if old_ckpt:
    cmd += ["--snet-ckpt", f"snet_buggy_target={old_ckpt}"]
if new_ckpt.exists():
    cmd += ["--snet-ckpt", f"snet_fixed_target={new_ckpt}"]
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError:
    traceback.print_exc()

report = {}
if (OUT / "step3_map.json").exists():
    report = json.loads((OUT / "step3_map.json").read_text())
report["retrain_seconds"] = train_seconds
hist = OUT / "snet" / "joint_fixed_target" / "history.json"
if hist.exists():
    h = json.loads(hist.read_text())["history"]
    report["retrain_ball_f1_at_4px"] = [
        e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e
    ]
    report["retrain_val_loss"] = h[-1].get("val_loss")
(OUT / "step3_summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:3000], flush=True)
print("\ndone", flush=True)
