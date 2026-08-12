"""End-to-end player position error, in metres on the pitch.

Every number this project has produced so far is a component score — ball F1,
detection mAP, landmark error in pixels — and none of them is the quantity the
architecture exists to produce. GS-HOTA measures **metres on the pitch**, with a
5 m tolerance, and metres is what a minimap consumes. This closes that gap for the
localisation half of the task. It is deliberately *not* GS-HOTA: that also scores
role, team and jersey, and there are no heads for those.

The point of the module is the decomposition. Running the full pipeline gives one
number and no diagnosis, so each stage is also run against ground truth for the
other:

| configuration | isolates |
|---|---|
| GT boxes + GT homography | the harness itself; should be ~0 |
| GT boxes + **predicted** homography | the pitch head |
| **predicted** boxes + GT homography | the detector |
| predicted + predicted | the pipeline as it would ship |

Section 6.6 measured the budget this is scored against: with 8 or more keypoints,
sigma = 3 px of keypoint noise costs 0.39 m median and puts 2.1% of players beyond
5 m. Those are the numbers to beat.

The ground-truth homography is fitted from athlete foot points against their
annotated pitch coordinates — the same construction section 6.6 validated at a
median residual of 0.09 m.
"""

from __future__ import annotations

import numpy as np

from soccernet_tracking_edge.core.pitch import LANDMARKS, fit_homography, project
from soccernet_tracking_edge.core.targets import soft_argmax

GS_HOTA_TOLERANCE_M = 5.0
MIN_KEYPOINTS = 4

# Tuned by sweeping both against metres on the valid split, not chosen by taste.
# At 0.5/2.0 the pipeline left 22.9% of players unusable; at 0.15/4.0 it leaves
# 11.7%. The optimum is real rather than monotone: dropping to 0.08 recovers more
# landmarks (11 vs 9) and nearly eliminates frames with no homography (1.0% vs
# 4.0%), but the extra detections are noisy enough to worsen the fit anyway
# (12.8% unusable). A looser RANSAC wins at every threshold, which is what
# admitting weaker detections should require.
DEFAULT_KP_THRESHOLD = 0.15
DEFAULT_RANSAC_M = 4.0
LANDMARK_PITCH = np.array(list(LANDMARKS.values()), dtype=np.float64)


def decode_landmarks(
    heat: np.ndarray, kp_stride: int, scale_x: float, scale_y: float,
    threshold: float = DEFAULT_KP_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel soft-argmax into native image pixels.

    Returns ``(points, found)`` where ``found[k]`` says whether landmark *k*
    produced a peak. Channel index is landmark index, so a channel that stays
    silent must leave a hole rather than shift its neighbours.
    """
    n = heat.shape[0]
    points = np.zeros((n, 2), dtype=np.float64)
    found = np.zeros(n, dtype=bool)
    for k in range(n):
        got = soft_argmax(heat[k], threshold=threshold)
        if got is None:
            continue
        points[k] = (got[0] * kp_stride / scale_x, got[1] * kp_stride / scale_y)
        found[k] = True
    return points, found


def homography_from_landmarks(
    points: np.ndarray, found: np.ndarray, ransac_m: float = DEFAULT_RANSAC_M
) -> np.ndarray | None:
    """Image -> pitch from whichever landmarks the head emitted.

    RANSAC because a keypoint head will occasionally confuse one landmark for
    another, and a single swapped correspondence wrecks a least-squares fit. The
    threshold is in **metres** — OpenCV measures reprojection error in the
    destination space, which here is the pitch.
    """
    if found.sum() < MIN_KEYPOINTS:
        return None
    try:
        h, _ = fit_homography(points[found], LANDMARK_PITCH[found], ransac_m=ransac_m)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return h


def gt_homography(anns: list[dict]) -> np.ndarray | None:
    """Reference homography from athlete foot points against their pitch coords."""
    feet, pitch = [], []
    for a in anns:
        xy = a.get("pitch_xy")
        if a["category_id"] == 0 or not xy or xy[0] is None or xy[1] is None:
            continue
        bx, by, bw, bh = a["bbox"]
        feet.append([bx + bw / 2.0, by + bh])
        pitch.append(xy)
    if len(feet) < 8:
        return None
    try:
        h, _ = fit_homography(np.asarray(feet), np.asarray(pitch), ransac_m=1.0)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return h


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xywh box against many."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0 = b[:, 0], b[:, 1]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + (b[:, 2] * b[:, 3]) - inter + 1e-9)


def match_boxes(
    pred: list[dict], gt: list[dict], iou_threshold: float = 0.5
) -> list[tuple[dict, dict]]:
    """Greedy IoU matching, highest-confidence prediction first.

    Only matched pairs contribute a position error. Unmatched predictions and
    unmatched ground truth are a *detection* failure, already measured by mAP, and
    folding them in here would conflate two questions.
    """
    if not pred or not gt:
        return []
    gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float64)
    taken = np.zeros(len(gt), dtype=bool)
    pairs = []
    for p in sorted(pred, key=lambda d: -d.get("score", 0.0)):
        ious = _iou(np.asarray(p["bbox"], dtype=np.float64), gt_boxes)
        ious[taken] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= iou_threshold:
            taken[j] = True
            pairs.append((p, gt[j]))
    return pairs


def foot_points(boxes: list[dict]) -> np.ndarray:
    b = np.array([d["bbox"] for d in boxes], dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3]], axis=1)


def position_errors(
    homography: np.ndarray, image_pts: np.ndarray, truth_pitch: np.ndarray
) -> np.ndarray:
    """Metres between projected image points and their annotated pitch positions."""
    if len(image_pts) == 0:
        return np.zeros(0)
    got = project(homography, image_pts)
    return np.linalg.norm(got - truth_pitch, axis=1)


def summarise(errors: np.ndarray, label: str = "") -> dict:
    e = np.asarray(errors, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(e.size),
        "median_m": round(float(np.median(e)), 3),
        "p90_m": round(float(np.percentile(e, 90)), 3),
        # The quantity GS-HOTA actually cares about: beyond the tolerance a
        # detection scores essentially zero however good the box was.
        "pct_over_5m": round(100.0 * float((e > GS_HOTA_TOLERANCE_M).mean()), 2),
        "pct_under_1m": round(100.0 * float((e <= 1.0).mean()), 2),
    }


# --- temporal smoothing -----------------------------------------------------

# A coarse grid over the pitch, used as the representation the smoothing acts on.
# Corners alone would be badly conditioned; a spread grid constrains the whole
# frame and keeps the re-fit stable.
_SMOOTH_GRID = np.array(
    [[x, y] for x in (-52.5, -26.0, 0.0, 26.0, 52.5) for y in (-34.0, 0.0, 34.0)],
    dtype=np.float64,
)


def smooth_homographies(
    homographies: list[np.ndarray | None], window: int = 5
) -> list[np.ndarray | None]:
    """Temporally filter a sequence of per-frame homographies.

    **Not elementwise on the matrices.** A homography has nine entries, is defined
    only up to scale, and averaging them is meaningless — small changes in h[2, :]
    move the projection enormously. Instead each frame's homography is turned into
    the image positions of a fixed grid of pitch points, those *trajectories* are
    filtered, and the homography is re-fitted from the filtered correspondences.
    That representation is well conditioned and every quantity in it is a pixel.

    **Median, not mean.** The measured error profile is a tail: the median frame is
    already good (0.76 m) and a minority collapse (p90 8.63 m). A mean or
    Savitzky-Golay filter would smear a catastrophic frame across its neighbours;
    a median filter rejects it outright, which is the whole point.

    Frames with no homography (fewer than four landmarks) enter as gaps and are
    filled from their neighbours, which is the repair this exists for.

    The cost is lag on a moving camera: a window of ``w`` frames at 25 fps spans
    ``w / 25`` seconds, and a broadcast camera pans continuously. Window size
    trades tail repair against that lag and is meant to be swept, not assumed.
    """
    n = len(homographies)
    if n == 0:
        return []
    half = max(1, window // 2)

    # pitch -> image for every frame, as (n, points, 2) with NaN for missing frames
    tracks = np.full((n, len(_SMOOTH_GRID), 2), np.nan, dtype=np.float64)
    for t, h in enumerate(homographies):
        if h is None:
            continue
        try:
            tracks[t] = project(np.linalg.inv(h), _SMOOTH_GRID)
        except np.linalg.LinAlgError:
            continue

    out: list[np.ndarray | None] = []
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        window_pts = tracks[lo:hi]
        valid = ~np.isnan(window_pts[:, :, 0])
        if valid.sum(axis=0).min() == 0:
            # No neighbour saw some grid point; nothing to interpolate from.
            out.append(homographies[t])
            continue
        smoothed = np.nanmedian(window_pts, axis=0)
        if not np.isfinite(smoothed).all():
            out.append(homographies[t])
            continue
        try:
            h, _ = fit_homography(smoothed, _SMOOTH_GRID, ransac_m=DEFAULT_RANSAC_M)
        except (ValueError, np.linalg.LinAlgError):
            h = homographies[t]
        out.append(h)
    return out
