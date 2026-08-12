"""Sweep the landmark decode threshold, scored in metres.

The metres decomposition put the whole end-to-end error in the pitch head's tail,
and located the cause in landmark *count* rather than per-landmark precision: the
median frame yields 7 landmarks against the 8 that section 6.6 identified as the
safe floor, and 11.4% of frames yield fewer than four and produce no homography at
all.

The decode threshold is the cheapest lever on that. A landmark recovered at low
confidence still constrains the fit; a missing one does not, and RANSAC is already
there to reject the ones that come back wrong. This is inference-only — no
training — so it costs minutes.

The known ceiling, stated up front so the result is read correctly: section 6.6
measured a median of 9 landmarks actually *visible* per frame out of 33, and the
head already finds 7 of them. Lowering the threshold can recover about two. That
crosses the k=8 floor and does not move the frame far from it. The substantive fix
is a larger landmark set, and this sweep is meant to size that decision, not to
replace it.

RANSAC threshold is swept alongside, because admitting weaker detections changes
how tolerant the fit should be.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
THRESHOLDS = [0.5, 0.35, 0.25, 0.15, 0.08]
RANSAC_M = [2.0, 4.0]


def sh(*cmd, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run([str(c) for c in cmd], check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (non-fatal: {exc})", flush=True)


SCRATCH.mkdir(parents=True, exist_ok=True)
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

# The stride-4 checkpoint: stride 2 was already measured worse in metres.
hits = sorted(Path("/kaggle/input").rglob("cap_w32/best.pt"))
if not hits:
    raise SystemExit("cap_w32 checkpoint not mounted")
ckpt = hits[0]
print(f"checkpoint: {ckpt}", flush=True)

rows = []
for thr in THRESHOLDS:
    for rm in RANSAC_M:
        out = OUT / f"metres_t{thr}_r{rm}.json"
        subprocess.run([
            sys.executable, str(scripts_dir / "eval_metres.py"),
            "--ckpt", str(ckpt),
            "--gsr", str(OUT / "gsr"), "--frames", str(SCRATCH / "gsr"),
            "--split", "valid", "--val-stride", "87", "--val-limit", "500",
            "--kp-threshold", str(thr), "--ransac-m", str(rm), "--out", str(out),
        ], check=False)
        if not out.exists():
            continue
        r = json.loads(out.read_text())
        full = r["results"].get("pred_boxes_pred_H", {})
        pitch_only = r["results"].get("gt_boxes_pred_H", {})
        no_h = r["frames_without_predicted_homography"]
        frames = r["frames"]
        # Frames with no homography are excluded from the error statistics, so the
        # honest headline combines both failure modes.
        unusable = 1 - (1 - no_h / frames) * (1 - full.get("pct_over_5m", 0) / 100)
        rows.append({
            "threshold": thr, "ransac_m": rm,
            "landmarks_median": r["landmarks_found_median"],
            "no_homography_pct": round(100 * no_h / frames, 1),
            "pitch_median_m": pitch_only.get("median_m"),
            "pitch_p90_m": pitch_only.get("p90_m"),
            "full_median_m": full.get("median_m"),
            "full_p90_m": full.get("p90_m"),
            "full_over5_pct": full.get("pct_over_5m"),
            "unusable_pct": round(100 * unusable, 1),
        })
        print(f"  thr={thr} ransac={rm}m -> landmarks {rows[-1]['landmarks_median']}, "
              f"no-H {rows[-1]['no_homography_pct']}%, p90 {rows[-1]['full_p90_m']}m, "
              f"unusable {rows[-1]['unusable_pct']}%", flush=True)

(OUT / "threshold_sweep.json").write_text(json.dumps({
    "checkpoint": str(ckpt),
    "baseline": {"threshold": 0.5, "ransac_m": 2.0, "unusable_pct": 22.9},
    "rows": rows,
}, indent=2))

print("\n=== sweep (lower unusable% is better)", flush=True)
hdr = f"{'thr':>5} {'ransac':>7} {'landmarks':>10} {'no-H%':>7} {'p90 m':>8} {'>5m%':>7} {'unusable%':>10}"
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: x["unusable_pct"]):
    print(f"{r['threshold']:>5} {r['ransac_m']:>7} {r['landmarks_median']:>10} "
          f"{r['no_homography_pct']:>7} {r['full_p90_m']:>8} {r['full_over5_pct']:>7} "
          f"{r['unusable_pct']:>10}")
print("\ndone", flush=True)
