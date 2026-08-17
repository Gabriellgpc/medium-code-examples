"""One landmark arm, six epochs, DataParallel off. Push one copy per arm.

Section 9.17 identified `nn.DataParallel` as the leak that killed three runs at
9 GB per epoch, so this is the first configuration that can actually reach the end
of a schedule. Section 9.16 is why both arms run rather than only the new one: the
33-landmark reference this experiment is judged against (8.0% of players beyond
5 m) was measured under DataParallel, which put 4 samples per GPU through
BatchNorm with per-device statistics. Unwrapped it is 8 in one pass. That is a
better configuration and not a comparable one, so the old reference cannot be
reused and the control has to be re-run alongside.

LANDMARK_SET is substituted per kernel. Everything else is held fixed at what the
measurements have settled: w32 (section 9.8, AP75 +13.7% and free to train),
kp_stride 4 (section 9.11, stride 2 was better in pixels and worse in metres),
batch 8 and lr 1e-3 unchanged since step 1.

Six epochs at roughly 40 min each — section 9.9 measured DataParallel as a 0.81x
throughput *loss*, so removing it should also shorten the epoch from the 50 min
attempt 3 spent. About 4 hours, inside Kaggle's 9-hour session limit with room to
spare.

`--resume auto` is passed even though the OOM should be gone: if this dies for any
other reason, the rerun continues from the last completed epoch instead of paying
for the whole schedule twice. On a fresh run directory it is a no-op.
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
LANDMARK_SET = "expanded"  # substituted per kernel: expanded | base
SEED = 0            # substituted per kernel


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

TAG = f"{LANDMARK_SET}_w32_dp0_s{SEED}"
print(f"\n{'=' * 60}\n=== {TAG}: {LANDMARK_SET} landmarks, w32, {EPOCHS} epochs, "
      f"DataParallel OFF, seed {SEED}\n{'=' * 60}", flush=True)
t0 = time.time()
try:
    subprocess.run([
        sys.executable, str(scripts_dir / "train_snet.py"),
        "--heads", "ball,detection,pitch", "--landmark-set", LANDMARK_SET,
        "--trunk-width", "32", "--epochs", str(EPOCHS), "--batch", "8", "--lr", "1e-3",
        "--val-stride", "29", "--val-limit", "1500",
        "--workers", "2", "--eval-every", "2", "--tag", TAG, "--resume", "auto",
        "--seed", str(SEED),
    ], check=True)
except subprocess.CalledProcessError:
    traceback.print_exc()
train_seconds = round(time.time() - t0, 1)

report = {"train_seconds": train_seconds, "epochs": EPOCHS, "seed": SEED,
          "landmark_set": LANDMARK_SET, "data_parallel": False}
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
    # The number section 9.17 predicts stays flat near 2.6 GB. If it climbs, the
    # diagnosis was wrong and the trace says so without another kernel.
    report["rss_gb"] = [e.get("rss_gb") for e in h]
    report["kendall"] = [e.get("weights") for e in h][-1:]

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

(OUT / f"converged_{LANDMARK_SET}_s{SEED}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:3500], flush=True)
print("\ndone", flush=True)
