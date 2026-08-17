"""Ball decoding threshold sweep on the converged checkpoint. Minutes, not hours.

The converged ball head fires in only 42% of the frames that contain a ball, and
FN stays at 669 across every tolerance from 1 to 12 px — the signature of a decoder
returning nothing, not of a head aiming badly. `soft_argmax`'s 0.5 threshold came
from WASB and was never measured on this model.

Section 9.12 asked the same question of the pitch head and halved the unusable
fraction for free. This asks it of the ball before any training change is funded.

Only the valid split is prepared, and the checkpoint arrives by mounting the
`snet-conv-expanded-s1` kernel's output rather than re-training anything.
"""

import json
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
sh("nvidia-smi", "--query-gpu=name", "--format=csv", check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru", "albumentations")


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker}")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("ball_threshold_sweep.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

ckpts = sorted(Path("/kaggle/input").rglob("expanded_w32_dp0_s1/last.pt"))
if not ckpts:
    ckpts = sorted(Path("/kaggle/input").rglob("last.pt"))
if not ckpts:
    raise SystemExit("no checkpoint found -- is the training kernel attached?")
ckpt = ckpts[0]
print(f"checkpoint: {ckpt}", flush=True)

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["valid"]
t0 = time.time()
root = prepare_gsr.fetch()
if not (prepare_gsr.OUT / "valid" / "detection.json").exists():
    prepare_gsr.prepare(root / "valid", "valid")
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

out = OUT / "ball_threshold_sweep.json"
subprocess.run([
    sys.executable, str(scripts_dir / "ball_threshold_sweep.py"),
    "--ckpt", str(ckpt), "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"),
    "--split", "valid", "--val-stride", "29", "--val-limit", "1500",
    "--out", str(out),
], check=False)

if out.exists():
    print("\n=== summary ===", flush=True)
    print(json.dumps(json.loads(out.read_text())["results"]["4.0"], indent=1)[:2000],
          flush=True)
print("\ndone", flush=True)
