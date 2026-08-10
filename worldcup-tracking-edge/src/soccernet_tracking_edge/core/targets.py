"""Ground-truth heatmap builders.

Every head in SNet regresses a Gaussian heatmap, but the three heads want
*different* Gaussians, and the differences are not cosmetic:

* **Ball** uses WASB's real-valued map: a narrow bump whose non-zero minimum is
  pinned to ``c_min`` at radius ``d``. The point is that a binary disk makes the
  exact centre unrecoverable — every pixel inside the disk says "ball here" with
  equal confidence — so a model trained on it produces blurry peaks and imprecise
  localisation [WASB, Eq. 2].
* **Detection** uses CornerNet/CenterNet's size-adaptive Gaussian: a big player
  tolerates a bigger centre error than a small one, so sigma scales with the box.
* **Pitch** uses a fixed-width Gaussian, because a line intersection has no scale.

Coordinates coming in are in *native* pixels; ``scale`` maps them to the output
heatmap grid, and the caller is responsible for it being the model's real output
stride. Getting that wrong silently trains the model on shifted targets.
"""

from __future__ import annotations

import numpy as np

# WASB's published settings: d = 2.5 heatmap px, non-zero minimum 0.7.
BALL_RADIUS_PX = 2.5
BALL_C_MIN = 0.7


def ball_gaussian(
    heatmap: np.ndarray, cx: float, cy: float, d: float = BALL_RADIUS_PX,
    c_min: float = BALL_C_MIN,
) -> np.ndarray:
    """WASB's real-valued ball target, drawn in place.

    ``C`` is fixed by requiring the value at radius ``d`` to equal ``c_min``:
    ``C * exp(-1) = c_min``, hence ``C = c_min * e``. Values are clipped at 1, so
    the peak is a small flat plateau rather than a single spike — which is what
    makes centre-of-heatmap recovery stable.
    """
    h, w = heatmap.shape
    x0, x1 = max(0, int(np.floor(cx - d))), min(w, int(np.ceil(cx + d)) + 1)
    y0, y1 = max(0, int(np.floor(cy - d))), min(h, int(np.ceil(cy + d)) + 1)
    if x0 >= x1 or y0 >= y1:
        return heatmap

    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    inside = dist2 <= d * d
    c = c_min * np.e
    vals = np.minimum(c * np.exp(-dist2 / (d * d)), 1.0) * inside
    np.maximum(heatmap[y0:y1, x0:x1], vals, out=heatmap[y0:y1, x0:x1])
    return heatmap


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CornerNet's radius: the largest centre error that still keeps IoU >= min_overlap.

    Three quadratics, one per way a shifted box can still overlap enough; the
    binding constraint is the smallest root.
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(b1**2 - 4 * a1 * c1, 0))) / (2 * a1)

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - np.sqrt(max(b2**2 - 4 * a2 * c2, 0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (-b3 + np.sqrt(max(b3**2 - 4 * a3 * c3, 0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Standard unnormalised Gaussian, peak 1.0, drawn in place with max()."""
    h, w = heatmap.shape
    radius = int(max(1, np.ceil(3 * sigma)))
    x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius + 1)
    y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return heatmap
    ys, xs = np.mgrid[y0:y1, x0:x1]
    vals = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))
    np.maximum(heatmap[y0:y1, x0:x1], vals, out=heatmap[y0:y1, x0:x1])
    return heatmap


def detection_targets(
    boxes: np.ndarray, classes: np.ndarray, out_h: int, out_w: int,
    n_classes: int, scale: float,
) -> dict[str, np.ndarray]:
    """CenterNet targets: class heatmaps, plus size and offset at the centre pixel.

    ``boxes`` is (N, 4) as xywh in native pixels. The offset target exists because
    the centre lands between output pixels once divided by the stride, and without
    it every prediction is quantised to the grid — the same quantisation argument
    that forces sub-pixel decoding at inference.
    """
    heat = np.zeros((n_classes, out_h, out_w), dtype=np.float32)
    size = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for (bx, by, bw, bh), cls in zip(boxes, classes, strict=True):
        sw, sh = bw * scale, bh * scale
        if sw <= 0 or sh <= 0:
            continue
        cx = (bx + bw / 2.0) * scale
        cy = (by + bh / 2.0) * scale
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < out_w and 0 <= iy < out_h):
            continue
        sigma = max(gaussian_radius(sh, sw) / 3.0, 0.6)
        draw_gaussian(heat[int(cls)], cx, cy, sigma)
        size[0, iy, ix], size[1, iy, ix] = sw, sh
        offset[0, iy, ix], offset[1, iy, ix] = cx - ix, cy - iy
        mask[iy, ix] = 1.0

    return {"heat": heat, "size": size, "offset": offset, "mask": mask}


def keypoint_targets(
    points: np.ndarray, visible: np.ndarray, out_h: int, out_w: int,
    n_keypoints: int, scale: float, sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-width Gaussians per landmark, plus a per-landmark presence mask.

    The mask matters: most landmarks are out of frame in any given broadcast crop
    (median 9 of 33 visible, TRAINING-DESIGN section 6.6). Supervising an absent
    landmark's channel as all-zero is correct — the model should say "not here" —
    but the mask lets the caller weight those channels differently if the
    all-zero majority starts drowning the signal.
    """
    heat = np.zeros((n_keypoints, out_h, out_w), dtype=np.float32)
    present = np.zeros(n_keypoints, dtype=np.float32)
    for k in range(min(n_keypoints, len(points))):
        if not visible[k]:
            continue
        cx, cy = points[k][0] * scale, points[k][1] * scale
        if not (0 <= cx < out_w and 0 <= cy < out_h):
            continue
        draw_gaussian(heat[k], cx, cy, sigma)
        present[k] = 1.0
    return heat, present


def soft_argmax(heatmap: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float] | None:
    """Centre of heatmap mass above a threshold: WASB's CoH decoding.

    Returns ``(x, y, score)`` in heatmap coordinates, or None when nothing clears
    the threshold. Plain argmax would quantise to the grid, which at our output
    stride costs about a third of the pitch head's whole error budget.
    """
    mask = heatmap >= threshold
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    w = heatmap[ys, xs].astype(np.float64)
    total = w.sum()
    return float((xs * w).sum() / total), float((ys * w).sum() / total), float(total)
