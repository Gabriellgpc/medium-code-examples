"""Kaggle session step 2: measure the three things that decide the architecture.

Run after ``prepare_gsr.py`` in the same session, while the frames are still in
scratch. Everything here is *measurement* — no training, no model. Each check
answers a question that TRAINING-DESIGN.md currently marks as unmeasured, and
each can invalidate part of the plan.

**M1 — player box geometry.** Decides whether detection could ever share the
high-resolution trunk with the ball and pitch heads. If players survive at the
ball head's input resolution, v2's single-backbone merge is worth attempting; if
they collapse, the two-network split is permanent.

**M2 — ball annotation convention.** BlurBall's gain comes from moving the label
from the leading edge of the motion-blur streak to its centre. That graft only
exists if SN-GSR labels the edge. This measures which one it does, on the fast
frames where the difference is visible at all, and writes crops so the number can
be checked by eye.

**M3 — keypoint error budget.** The real question is not "is our pitch head
accurate" but "how accurate must it be". Ground-truth homographies come free
here: SN-GSR gives every athlete both an image box and a pitch position in
metres, so a homography can be fitted per frame without any calibration model.
Inverting it projects the known pitch landmarks into the image, giving perfect
synthetic keypoints. Perturbing *those* by sigma pixels and re-fitting converts
keypoint noise into metres of player-position error — the same unit GS-HOTA
scores with a 5 m tolerance.

Outputs land in ``/kaggle/working/gsr/measurements/`` (small, persisted).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SCRATCH = Path("/kaggle/tmp/gsr")
OUT = Path("/kaggle/working/gsr")
MEAS = OUT / "measurements"

# Input resolutions worth costing out. WASB trains at 288x512; the rest bracket it.
CANDIDATE_RESOLUTIONS = [(288, 512), (384, 640), (512, 896), (640, 1088)]

# A detector needs a few pixels of object to work with. These are the reference
# lines we report against, not thresholds anyone has proven for this dataset.
HEIGHT_FLOORS_PX = [8, 12, 16, 24]

SIGMAS_PX = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
KEYPOINT_BUDGETS = [4, 6, 8, 12, None]  # None = every visible landmark

CLASS_NAMES = {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}


def pct(values: np.ndarray, qs=(1, 5, 10, 25, 50, 75, 90, 99)) -> dict:
    if len(values) == 0:
        return {}
    return {f"p{q}": round(float(np.percentile(values, q)), 2) for q in qs}


def install_package() -> None:
    """Put soccernet_tracking_edge on the path (same glob as prepare_gsr.py)."""
    for candidate in Path("/kaggle/input").glob("**/soccernet_tracking_edge/__init__.py"):
        sys.path.insert(0, str(candidate.parent.parent))
        return
    raise SystemExit("source package not found under /kaggle/input")


# ---------------------------------------------------------------- M1


def m1_box_geometry(det_json: Path) -> dict:
    """Box size distributions, and what they become at each candidate resolution.

    Reported in *native* pixels first, because that is the only scale-free fact;
    the per-resolution table is derived from it so the arithmetic is visible
    rather than baked in.
    """
    data = json.loads(det_json.read_text())
    sizes = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    by_class: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_shapes: set[tuple[int, int]] = set()

    for a in data["annotations"]:
        _, _, w, h = a["bbox"]
        by_class[a["category_id"]].append((w, h))
        frame_shapes.add(sizes[a["image_id"]])

    if len(frame_shapes) != 1:
        print(f"  ! frames are not all one size: {sorted(frame_shapes)}", flush=True)
    src_w, src_h = sorted(frame_shapes)[0]

    out: dict = {"source_frame": [src_w, src_h], "per_class": {}}
    for cid, name in CLASS_NAMES.items():
        wh = np.asarray(by_class.get(cid, []), dtype=np.float64)
        if len(wh) == 0:
            continue
        widths, heights = wh[:, 0], wh[:, 1]
        # For the ball the meaningful size is the longest side, not the height.
        longest = np.maximum(widths, heights)
        entry = {
            "n": int(len(wh)),
            "height_px": pct(heights),
            "width_px": pct(widths),
            "longest_side_px": pct(longest),
            "at_resolution": {},
        }
        for rh, rw in CANDIDATE_RESOLUTIONS:
            scale = min(rw / src_w, rh / src_h)
            scaled_h = heights * scale
            scaled_long = longest * scale
            entry["at_resolution"][f"{rh}x{rw}"] = {
                "scale": round(float(scale), 4),
                "height_px": pct(scaled_h, qs=(1, 5, 10, 50, 90)),
                "longest_side_px": pct(scaled_long, qs=(1, 5, 10, 50, 90)),
                "pct_below": {
                    f"{f}px": round(100.0 * float((scaled_h < f).mean()), 2)
                    for f in HEIGHT_FLOORS_PX
                },
            }
        out["per_class"][name] = entry
    return out


# ---------------------------------------------------------------- M2


def _blur_streak(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Centroid, unit principal axis and half-length of the bright blob.

    The ball is the bright, desaturated thing on green grass, so 'bright minus
    saturated' separates it without needing a trained model. Returns None when no
    blob is found, which is itself informative and gets counted.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    score = val - sat
    thr = float(score.mean() + 1.5 * score.std())
    mask = (score > thr).astype(np.uint8)
    if mask.sum() < 4:
        return None

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    centre = np.array([crop.shape[1] / 2.0, crop.shape[0] / 2.0])
    # The annotation sits at the crop centre by construction, so the component we
    # want is the one nearest it — not the largest, which is often a shirt.
    best = min(
        range(1, n),
        key=lambda i: np.linalg.norm(centroids[i] - centre) - 0.05 * stats[i, cv2.CC_STAT_AREA],
    )
    pts = np.column_stack(np.nonzero(labels == best))[:, ::-1].astype(np.float64)
    if len(pts) < 4:
        return None

    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    half_len = float(np.abs(centred @ axis).max())
    return centroid, axis, half_len


def m2_ball_convention(
    track_json: Path, root: Path, n_samples: int = 240, min_disp_px: float = 12.0
) -> dict:
    """Is the ball labelled at the centre of its blur streak, or at one end?

    Only fast frames can answer this: with no motion there is no streak and both
    conventions coincide. Displacement between consecutive annotated frames is
    the speed proxy.
    """
    data = json.loads(track_json.read_text())
    fast: list[tuple[str, dict, np.ndarray]] = []

    for seq in data["sequences"]:
        frames = seq["frames"]
        for i in range(1, len(frames)):
            a, b = frames[i - 1], frames[i]
            if not (a["visible"] and b["visible"]):
                continue
            motion = np.array(b["xy"]) - np.array(a["xy"])
            if np.linalg.norm(motion) >= min_disp_px:
                fast.append((seq["sequence"], b, motion))

    rng = random.Random(0)
    rng.shuffle(fast)
    sample = fast[:n_samples]
    print(f"  {len(fast)} fast frames (>{min_disp_px}px), sampling {len(sample)}", flush=True)

    offsets, angles, aspects, speeds, sizes = [], [], [], [], []
    contact: list[np.ndarray] = []
    no_blob = 0

    for _seq, frame, motion in sample:
        img_path = root / frame["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cx, cy = frame["xy"]
        speed = float(np.linalg.norm(motion))
        half = int(max(24, 1.5 * speed))
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = int(cx) + half, int(cy) + half
        if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
            continue
        crop = img[y0:y1, x0:x1]

        found = _blur_streak(crop)
        if found is None:
            no_blob += 1
            continue
        centroid, axis, half_len = found
        if half_len < 2.0:
            continue

        unit = motion / speed
        # Sign the axis so it points the way the ball is travelling; a principal
        # axis is direction-agnostic and would otherwise scramble the offset sign.
        if axis @ unit < 0:
            axis = -axis

        annotated = np.array([crop.shape[1] / 2.0, crop.shape[0] / 2.0])
        # +1 means the label sits at the leading tip, -1 at the trailing tip,
        # 0 at the centre of the streak.
        offsets.append(float(((annotated - centroid) @ unit) / half_len))
        angles.append(float(np.degrees(np.arccos(np.clip(abs(axis @ unit), 0, 1)))))
        speeds.append(speed)
        sizes.append(frame["size"])
        if frame["size"]:
            aspects.append(float(frame["size"]))
        if len(contact) < 24:
            vis = crop.copy()
            cv2.circle(vis, tuple(annotated.astype(int)), 3, (0, 0, 255), 1)
            cv2.circle(vis, tuple(centroid.astype(int)), 3, (0, 255, 0), 1)
            contact.append(cv2.resize(vis, (96, 96), interpolation=cv2.INTER_NEAREST))

    if contact:
        rows = [np.hstack(contact[i : i + 6]) for i in range(0, len(contact) - 5, 6)]
        if rows:
            MEAS.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(MEAS / "m2_ball_crops.png"), np.vstack(rows))

    off = np.asarray(offsets)
    corr = (
        float(np.corrcoef(speeds, sizes)[0, 1])
        if len(speeds) > 2 and None not in sizes
        else None
    )
    return {
        "sampled": len(sample),
        "measured": int(len(off)),
        "no_blob_found": no_blob,
        # Near 0 => labelled at blur centre (nothing for BlurBall's graft to fix).
        # Near +1 => leading edge; near -1 => trailing edge.
        "offset_along_motion": pct(off, qs=(10, 25, 50, 75, 90)) if len(off) else {},
        "offset_mean": round(float(off.mean()), 3) if len(off) else None,
        "streak_vs_motion_angle_deg": pct(np.asarray(angles), qs=(50, 90)) if angles else {},
        "speed_px": pct(np.asarray(speeds), qs=(50, 90)) if speeds else {},
        # If the annotated box grows with speed, the label already covers the
        # whole streak, which is a second way of answering the same question.
        "corr_speed_vs_boxsize": round(corr, 3) if corr is not None else None,
        "contact_sheet": "m2_ball_crops.png (red = annotation, green = streak centroid)",
    }


# ---------------------------------------------------------------- M3


def m3_keypoint_budget(det_json: Path, n_frames: int = 300) -> dict:
    """Convert keypoint noise in pixels into player-position error in metres."""
    from soccernet_tracking_edge.core.pitch import LANDMARKS, fit_homography, project

    data = json.loads(det_json.read_text())
    by_image: dict[int, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        if a.get("pitch_xy") and None not in a["pitch_xy"] and a["category_id"] != 0:
            by_image[a["image_id"]].append(a)
    images = {im["id"]: im for im in data["images"]}

    usable = [i for i, anns in by_image.items() if len(anns) >= 8]
    rng = random.Random(0)
    rng.shuffle(usable)
    usable = usable[:n_frames]
    print(f"  {len(usable)} frames with >=8 athletes carrying pitch coords", flush=True)

    names = list(LANDMARKS)
    model_pts = np.array([LANDMARKS[n] for n in names], dtype=np.float64)
    errors: dict[str, dict[int, list[float]]] = {
        f"{s}": defaultdict(list) for s in SIGMAS_PX
    }
    visible_counts, gt_residuals, skipped = [], [], 0
    noise_rng = np.random.default_rng(0)

    for img_id in usable:
        anns = by_image[img_id]
        im = images[img_id]
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float64)
        feet = np.stack([boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3]], axis=1)
        pitch = np.array([a["pitch_xy"] for a in anns], dtype=np.float64)

        try:
            # NOTE: fit_homography's `ransac_px` threshold is applied in the
            # *destination* space, which here is the pitch — so the unit is
            # metres, not pixels. The parameter name in core/pitch.py is
            # misleading and should be fixed separately.
            H_gt, res = fit_homography(feet, pitch, ransac_px=1.0)
            H_inv = np.linalg.inv(H_gt)
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1
            continue
        gt_residuals.append(float(np.median(res)))

        # Perfect synthetic keypoints: where each pitch landmark lands in this frame.
        lm_img = project(H_inv, model_pts)
        inside = (
            (lm_img[:, 0] >= 0) & (lm_img[:, 0] < im["width"])
            & (lm_img[:, 1] >= 0) & (lm_img[:, 1] < im["height"])
        )
        n_vis = int(inside.sum())
        visible_counts.append(n_vis)
        if n_vis < 4:
            continue
        vis_img, vis_model = lm_img[inside], model_pts[inside]
        truth = project(H_gt, feet)

        for sigma in SIGMAS_PX:
            for budget in KEYPOINT_BUDGETS:
                k = n_vis if budget is None else budget
                if k > n_vis:
                    continue
                idx = noise_rng.choice(n_vis, size=k, replace=False)
                noisy = vis_img[idx] + noise_rng.normal(0.0, sigma, size=(k, 2))
                try:
                    H_s, _ = fit_homography(noisy, vis_model[idx])
                    got = project(H_s, feet)
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    continue
                d = np.linalg.norm(got - truth, axis=1)
                if np.isfinite(d).all():
                    errors[f"{sigma}"][k].extend(d.tolist())

    table: dict = {}
    for sigma, per_k in errors.items():
        table[sigma] = {}
        for k, vals in sorted(per_k.items()):
            v = np.asarray(vals)
            table[sigma][f"k={k}"] = {
                "median_m": round(float(np.median(v)), 3),
                "p90_m": round(float(np.percentile(v, 90)), 3),
                # GS-HOTA's LocSim decays with a 5 m tolerance; beyond it a
                # detection scores essentially zero however good the box was.
                "pct_over_5m": round(100.0 * float((v > 5.0).mean()), 2),
            }
    return {
        "frames_used": len(usable),
        "frames_skipped": skipped,
        "gt_fit_residual_m": pct(np.asarray(gt_residuals), qs=(50, 90)) if gt_residuals else {},
        "visible_landmarks": pct(np.asarray(visible_counts), qs=(10, 25, 50, 75, 90)),
        "player_error_m": table,
    }


# ---------------------------------------------------------------- driver


def main(split: str = "train") -> None:
    install_package()
    MEAS.mkdir(parents=True, exist_ok=True)
    det = OUT / split / "detection.json"
    track = OUT / split / "ball_track.json"
    root = SCRATCH / split
    for p in (det, track):
        if not p.exists():
            raise SystemExit(f"missing {p} — run prepare_gsr.py first")

    print("M1: box geometry", flush=True)
    m1 = m1_box_geometry(det)
    print("M2: ball annotation convention", flush=True)
    m2 = m2_ball_convention(track, root)
    print("M3: keypoint error budget", flush=True)
    m3 = m3_keypoint_budget(det)

    report = {"split": split, "m1_box_geometry": m1, "m2_ball_convention": m2,
              "m3_keypoint_budget": m3}
    (MEAS / f"{split}_measurements.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"\nwritten to {MEAS}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
