"""Retrain with the expanded landmark set, then score it in metres.

The ceiling measurement said this is the change worth paying for: 41% of frames
sat below §6.6's eight-correspondence safety floor with 33 landmarks and 6% do with
47, and the 33-point set fails outright on 3.1% of frames even with *perfect*
keypoints — geometric degeneracy no model accuracy can repair.

It also produced a prediction to check this run against. The head currently
localises landmarks at 7-9 px; at sigma = 8 the geometry predicts 6.2% of players
beyond tolerance for 33 landmarks against a measured ~8%, so the model behaves
about as geometry says. The same row for 47 landmarks reads **1.4%**, so this run
should land the >5 m rate somewhere near 1.5-2%.

The risk in that prediction is specific: corners and line intersections are crisp,
while a sample point at 30 degrees on a circle has no local feature marking it and
must be inferred from the arc's shape. If the new landmarks are harder to localise,
the payoff shrinks. `evaluate_pitch` reports detection rate alongside error, so the
two failure modes stay separable.

Configuration follows what the measurements have settled: w32 (capacity won AP75
+13.7% and was free to train), kp_stride 4 (stride 2 was better in pixels and
substantially worse in metres), batch 8 and lr 1e-3 unchanged from every prior run.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
EPOCHS = 6


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

TAG = "expanded_w32"
print(f"\n{'=' * 60}\n=== {TAG}: 47 landmarks, w32, {EPOCHS} epochs\n{'=' * 60}", flush=True)
t0 = time.time()
try:
    subprocess.run([
        sys.executable, str(scripts_dir / "train_snet.py"),
        "--heads", "ball,detection,pitch", "--landmark-set", "expanded",
        "--trunk-width", "32", "--epochs", str(EPOCHS), "--batch", "8", "--lr", "1e-3",
        "--val-stride", "29", "--val-limit", "1500",
        "--workers", "4", "--eval-every", "2", "--tag", TAG,
    ], check=True)
except subprocess.CalledProcessError:
    traceback.print_exc()
train_seconds = round(time.time() - t0, 1)

report = {"train_seconds": train_seconds, "epochs": EPOCHS, "landmark_set": "expanded"}
hist = OUT / "snet" / TAG / "history.json"
if hist.exists():
    h = json.loads(hist.read_text())["history"]
    report["ball_f1"] = [e["ball_val"]["4"]["f1"] for e in h if "ball_val" in e]
    report["pitch"] = [
        {k: e["pitch_val"].get(k)
         for k in ("median_px", "p90_px", "pct_under_budget", "detected_fraction")}
        for e in h if "pitch_val" in e
    ]
    report["epoch_seconds"] = [e["seconds"] for e in h]
    report["peak_vram_gb"] = [e.get("peak_vram_gb") for e in h]
    report["kendall"] = [e.get("weights") for e in h][-1:]

# Both checkpoints: best_pitch is selected on the head this run is about, last is
# the annealed one that cross-run detection numbers should use.
for ck_name in ("best_pitch.pt", "last.pt"):
    ck = OUT / "snet" / TAG / ck_name
    if not ck.exists():
        continue
    out = OUT / f"metres_{ck_name.replace('.pt', '')}.json"
    subprocess.run([
        sys.executable, str(scripts_dir / "eval_metres.py"),
        "--ckpt", str(ck), "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"),
        "--split", "valid", "--val-stride", "87", "--val-limit", "500", "--out", str(out),
    ], check=False)
    if out.exists():
        report[f"metres_{ck_name}"] = json.loads(out.read_text())

(OUT / "expanded_summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:3500], flush=True)
print("\ndone", flush=True)
