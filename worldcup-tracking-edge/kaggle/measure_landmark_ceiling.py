"""Is the expanded landmark set worth retraining for? Measure before paying.

The residual error after threshold tuning and temporal smoothing is systematic:
frames where too few landmarks are visible, which neighbouring frames cannot repair
because they are the same shot. Expanding the landmark set is the indicated fix,
and it is the expensive one — new output channels and a full retrain.

So the ceiling gets measured first, from geometry alone. For each frame the
ground-truth homography is fitted from athlete foot points (§6.6 validated that
construction at a 0.09 m residual), both landmark sets are projected into the
frame, and the visible ones counted. No model, no inference, no training.

Then the §6.6 sensitivity is re-run on the *actual* visible sets: perturb the
landmarks a frame really shows by sigma pixels, re-fit, and measure the player
position error in metres. That converts "more landmarks" into the unit the project
is judged in, and it is what decides whether the retrain is worth its GPU hours.

A result of "median visible goes 9 -> 12" would argue against the expansion as
firmly as "9 -> 25" argues for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def locate_package() -> None:
    for c in Path("/kaggle/input").rglob("soccernet_tracking_edge/__init__.py"):
        sys.path.insert(0, str(c.parent.parent))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


locate_package()

from soccernet_tracking_edge.core.pitch import (  # noqa: E402
    EXPANDED_LANDMARKS,
    LANDMARKS,
    fit_homography,
    project,
)
from soccernet_tracking_edge.core.pitch_eval import gt_homography  # noqa: E402

SIGMAS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
FRAME_W, FRAME_H = 1920, 1080


def visible_mask(h_pitch2img: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray:
    img = project(h_pitch2img, pitch_pts)
    return (
        (img[:, 0] >= 0) & (img[:, 0] < FRAME_W)
        & (img[:, 1] >= 0) & (img[:, 1] < FRAME_H)
    ), img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsr", default="/kaggle/working/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--stride", type=int, default=29)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--out", default="/kaggle/working/landmark_ceiling.json")
    args = ap.parse_args()

    data = json.loads((Path(args.gsr) / args.split / "detection.json").read_text())
    from collections import defaultdict

    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    images = data["images"][:: args.stride][: args.limit]
    print(f"{len(images)} frames", flush=True)

    sets = {
        "base_33": np.array(list(LANDMARKS.values()), dtype=np.float64),
        "expanded_47": np.array(list(EXPANDED_LANDMARKS.values()), dtype=np.float64),
    }
    counts = {k: [] for k in sets}
    errors = {k: {s: [] for s in SIGMAS} for k in sets}
    rng = np.random.default_rng(0)
    used = 0

    for im in images:
        a = anns.get(im["id"], [])
        h_gt = gt_homography(a)
        if h_gt is None:
            continue
        used += 1
        h_inv = np.linalg.inv(h_gt)

        feet = np.stack([
            [x["bbox"][0] + x["bbox"][2] / 2.0, x["bbox"][1] + x["bbox"][3]]
            for x in a if x["category_id"] != 0 and x.get("pitch_xy")
            and x["pitch_xy"][0] is not None
        ])
        truth = project(h_gt, feet)

        for name, pitch_pts in sets.items():
            vis, img_pts = visible_mask(h_inv, pitch_pts)
            counts[name].append(int(vis.sum()))
            if vis.sum() < 4:
                continue
            for sigma in SIGMAS:
                noisy = img_pts[vis] + rng.normal(0.0, sigma, size=(int(vis.sum()), 2))
                try:
                    h, _ = fit_homography(noisy, pitch_pts[vis])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                d = np.linalg.norm(project(h, feet) - truth, axis=1)
                if np.isfinite(d).all():
                    errors[name][sigma] += d.tolist()

    def pct(v, q):
        return round(float(np.percentile(v, q)), 2) if len(v) else None

    report = {"frames": used, "visibility": {}, "sensitivity": {}}
    print("\n=== landmarks visible per frame")
    for name, c in counts.items():
        c = np.asarray(c)
        report["visibility"][name] = {
            "total_defined": len(sets[name]),
            "median": float(np.median(c)), "p10": pct(c, 10), "p90": pct(c, 90),
            "pct_frames_under_4": round(100.0 * float((c < 4).mean()), 2),
            "pct_frames_under_8": round(100.0 * float((c < 8).mean()), 2),
        }
        r = report["visibility"][name]
        print(f"  {name:<12} of {r['total_defined']:>3} defined: median {r['median']:>5.1f}  "
              f"p10 {r['p10']:>5.1f}  p90 {r['p90']:>5.1f}  "
              f"<4 in {r['pct_frames_under_4']:>5.2f}% of frames  "
              f"<8 in {r['pct_frames_under_8']:>5.2f}%")

    print("\n=== player position error, perturbing the landmarks each frame really shows")
    hdr = f"{'sigma px':>9} | " + " | ".join(f"{n:>26}" for n in sets)
    print(hdr)
    print("-" * len(hdr))
    for sigma in SIGMAS:
        cells = []
        for name in sets:
            e = np.asarray(errors[name][sigma])
            if len(e) == 0:
                cells.append(f"{'-':>26}")
                continue
            med, over = float(np.median(e)), 100.0 * float((e > 5.0).mean())
            report["sensitivity"].setdefault(str(sigma), {})[name] = {
                "median_m": round(med, 3), "p90_m": pct(e, 90),
                "pct_over_5m": round(over, 2),
            }
            cells.append(f"{med:8.3f} m  >5m {over:5.1f}%")
        print(f"{sigma:>9} | " + " | ".join(cells))

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
