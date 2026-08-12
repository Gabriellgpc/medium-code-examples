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
LANDMARK_PITCH = np.array(list(LANDMARKS.values()), dtype=np.float64)


def decode_landmarks(
    heat: np.ndarray, kp_stride: int, scale_x: float, scale_y: float,
    threshold: float = 0.5,
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
    points: np.ndarray, found: np.ndarray, ransac_m: float = 2.0
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
