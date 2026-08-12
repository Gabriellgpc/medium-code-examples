"""Metres on the pitch, for the two checkpoints that differ in output stride.

No training. This answers the question none of the per-head metrics answer — how
far off, in metres, is a player this pipeline places on the minimap — and it
answers it as a decomposition, so the number comes with a diagnosis.

Both checkpoints come from the same concurrent kernel at the same budget, so the
comparison also converts the kp_stride change from pixels into the unit that
matters: stride 2 improved landmark error from 8.78 to 7.09 px, and whether that
is worth its 29% loss of landmark detection can only be judged in metres.
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
sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", check=False)
sh(sys.executable, "-m", "pip", "install", "-q", "loguru", "albumentations")


def locate(marker: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(marker))
    if not hits:
        raise SystemExit(f"could not find {marker}")
    return hits[0].parent


pkg_dir = locate("soccernet_tracking_edge/__init__.py").parent
scripts_dir = locate("eval_metres.py")
sys.path.insert(0, str(pkg_dir))
sys.path.insert(0, str(scripts_dir))

import prepare_gsr  # noqa: E402

prepare_gsr.SPLITS = ["valid"]
t0 = time.time()
root = prepare_gsr.fetch()
if not (prepare_gsr.OUT / "valid" / "detection.json").exists():
    prepare_gsr.prepare(root / "valid", "valid")
print(f"data ready in {time.time() - t0:.0f}s", flush=True)

report = {}
for tag in ("cap_w32", "pitch_s2"):
    hits = sorted(Path("/kaggle/input").rglob(f"{tag}/best.pt"))
    if not hits:
        print(f"  {tag}: no checkpoint mounted", flush=True)
        continue
    print(f"\n{'=' * 60}\n=== {tag}  ({hits[0]})\n{'=' * 60}", flush=True)
    out = OUT / f"metres_{tag}.json"
    subprocess.run([
        sys.executable, str(scripts_dir / "eval_metres.py"),
        "--ckpt", str(hits[0]),
        "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"), "--split", "valid",
        "--val-stride", "87", "--val-limit", "500", "--out", str(out),
    ], check=False)
    if out.exists():
        report[tag] = json.loads(out.read_text())

(OUT / "metres_summary.json").write_text(json.dumps(report, indent=2))
print("\n=== summary", flush=True)
for tag, r in report.items():
    print(f"\n{tag}: kp_stride={r['kp_stride']}, "
          f"landmarks/frame median {r['landmarks_found_median']}, "
          f"no-homography frames {r['frames_without_predicted_homography']}")
    for k, v in r["results"].items():
        if v.get("n"):
            print(f"   {k:<20} median {v['median_m']:>6.2f} m  p90 {v['p90_m']:>6.2f}  "
                  f">5m {v['pct_over_5m']:>5.1f}%  <1m {v['pct_under_1m']:>5.1f}%")
print("\ndone", flush=True)
