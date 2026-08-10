"""Turn the detection head's heatmaps back into boxes.

CenterNet-style decoding: a 3x3 max-pool marks local maxima so no separate NMS is
needed, the top-K peaks become detections, and the size and offset maps at each
peak give the box. The offset is what recovers sub-pixel placement — without it
every box centre snaps to the output grid, which at our stride is 12 native pixels.

Everything returns boxes in **native image pixels**, because that is where the
ground truth lives and where mAP is defined. Decoding into network pixels and
comparing against native ground truth would silently inflate or deflate every IoU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _nms_peaks(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Zero everything that is not a local maximum in its kernel neighbourhood."""
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    return heat * (pooled == heat).float()


@torch.no_grad()
def decode_detections(
    heat: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    stride: int,
    scale_x: float,
    scale_y: float,
    top_k: int = 100,
    score_threshold: float = 0.01,
) -> list[list[dict]]:
    """Decode a batch of head outputs into per-image detection lists.

    ``heat`` is raw logits (B, C, H, W); ``scale_x``/``scale_y`` map network pixels
    back to native ones. ``score_threshold`` is deliberately low: mAP integrates
    over the precision-recall curve, so cutting low-confidence detections early
    throws away recall the metric would have credited.
    """
    scores = torch.sigmoid(heat)
    scores = _nms_peaks(scores)
    b, c, h, w = scores.shape

    flat = scores.view(b, -1)
    k = min(top_k, flat.shape[1])
    top_scores, top_idx = torch.topk(flat, k)

    cls = torch.div(top_idx, h * w, rounding_mode="floor")
    pix = top_idx % (h * w)
    ys = torch.div(pix, w, rounding_mode="floor")
    xs = pix % w

    size_flat = size.view(b, 2, -1)
    off_flat = offset.view(b, 2, -1)
    gathered_size = torch.gather(size_flat, 2, pix.unsqueeze(1).expand(-1, 2, -1))
    gathered_off = torch.gather(off_flat, 2, pix.unsqueeze(1).expand(-1, 2, -1))

    out: list[list[dict]] = []
    for i in range(b):
        dets = []
        for j in range(k):
            score = float(top_scores[i, j])
            if score < score_threshold:
                continue
            cx = (float(xs[i, j]) + float(gathered_off[i, 0, j])) * stride / scale_x
            cy = (float(ys[i, j]) + float(gathered_off[i, 1, j])) * stride / scale_y
            bw = float(gathered_size[i, 0, j]) * stride / scale_x
            bh = float(gathered_size[i, 1, j]) * stride / scale_y
            if bw <= 0 or bh <= 0:
                continue
            dets.append({
                "class": int(cls[i, j]),
                "score": score,
                "bbox": [cx - bw / 2.0, cy - bh / 2.0, bw, bh],   # COCO xywh
            })
        out.append(dets)
    return out


def to_coco_results(
    dets_per_image: list[list[dict]], image_ids: list[int], class_map: dict[int, int]
) -> list[dict]:
    """COCO detection records, with head class indices mapped back to dataset ids.

    ``class_map`` inverts the contiguous head indices used in training, so the
    numbers written here line up with the category ids in the ground-truth file.
    Getting this backwards produces a plausible-looking mAP of roughly zero.
    """
    results = []
    for dets, image_id in zip(dets_per_image, image_ids, strict=True):
        for d in dets:
            results.append({
                "image_id": int(image_id),
                "category_id": int(class_map[d["class"]]),
                "bbox": [round(float(v), 2) for v in d["bbox"]],
                "score": round(float(d["score"]), 5),
            })
    return results


def boxes_to_coco_results(
    xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray,
    image_id: int, class_map: dict[int, int],
) -> list[dict]:
    """Same, for a detector that already emits xyxy boxes (RF-DETR)."""
    out = []
    for (x0, y0, x1, y1), score, cls in zip(xyxy, scores, classes, strict=True):
        cid = class_map.get(int(cls))
        if cid is None:
            continue
        out.append({
            "image_id": int(image_id),
            "category_id": int(cid),
            "bbox": [round(float(x0), 2), round(float(y0), 2),
                     round(float(x1 - x0), 2), round(float(y1 - y0), 2)],
            "score": round(float(score), 5),
        })
    return out
