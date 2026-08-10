"""The converged run: all three heads, full split, a real training budget.

Everything before this was bounded on purpose — three epochs, enough to answer
relative questions (does sharing hurt, what did the bug cost, how does the head
compare to RF-DETR) without spending a week of quota. This run asks the absolute
question the others deliberately did not: what does the architecture reach.

The pitch head joins here for the first time. It was deferred because its target
was 33 channels at full resolution, 32 MB per sample; rasterising at trunk stride
instead costs 2.0 MB and still decodes to a median 0.66 native px, inside the
budget. That change also removed the pitch decoder, which took the whole model
from 30.6 ms to 20.7 ms on the target iGPU.

Trunk width stays at 18. w32 is now affordable (25.1 ms, 39.9 FPS) and is the
obvious next lever, but changing capacity and training budget in the same run
would leave neither attributable.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
EPOCHS = 12
BATCH = 8


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

print(f"\n{'=' * 60}\n=== converged: ball + detection + pitch, {EPOCHS} epochs\n{'=' * 60}",
      flush=True)
t0 = time.time()
try:
    subprocess.run([
        sys.executable, str(scripts_dir / "train_snet.py"),
        "--heads", "ball,detection,pitch",
        "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--val-stride", "29", "--val-limit", "1500",
        "--workers", "4", "--eval-every", "2",
        "--tag", "converged_all_heads",
    ], check=True)
except subprocess.CalledProcessError:
    traceback.print_exc()
train_seconds = round(time.time() - t0, 1)

# Score the finished model the same way Step 3 scored the bounded one.
ck = OUT / "snet" / "converged_all_heads" / "best.pt"
if ck.exists():
    print(f"\n{'=' * 60}\n=== mAP on the same 500 frames as Step 3\n{'=' * 60}", flush=True)
    subprocess.run([
        sys.executable, str(scripts_dir / "eval_step3.py"),
        "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
        "--val-stride", "87", "--val-limit", "500", "--skip-rfdetr",
        "--snet-ckpt", f"snet_converged={ck}",
        "--out", str(OUT / "converged_map.json"),
    ], check=False)

report = {"train_seconds": train_seconds, "epochs": EPOCHS}
hist = OUT / "snet" / "converged_all_heads" / "history.json"
if hist.exists():
    h = json.loads(hist.read_text())["history"]
    report["ball_f1_at_4px"] = [e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e]
    report["val_loss_per_eval"] = [e["val_loss"] for e in h if "val_loss" in e]
    report["epoch_seconds"] = [e["seconds"] for e in h]
    report["kendall_weights"] = [e.get("weights") for e in h][-1:]
if (OUT / "converged_map.json").exists():
    report["map"] = json.loads((OUT / "converged_map.json").read_text())["results"]
(OUT / "converged_summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:3000], flush=True)
print("\ndone", flush=True)
