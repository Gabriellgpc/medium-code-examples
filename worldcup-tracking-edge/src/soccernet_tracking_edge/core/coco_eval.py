"""COCO mAP, via pycocotools rather than a re-implementation.

mAP looks simple and is not: 101-point interpolated precision, per-area-range
breakdowns, maxDets caps, crowd handling, and a specific matching order. A
hand-rolled version that is 3% off is worse than no number at all, because it
looks right. So this wraps the reference implementation and only adds the plumbing
our data needs.

The plumbing that matters: our ground truth carries a ``ball`` category that the
detection head does not predict (the ball has its own full-resolution head), so
comparisons must be restricted to the categories both models actually emit.
Leaving the ball in would score our detector as missing every ball in the dataset.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

COCO_STATS = [
    ("mAP", "AP @ IoU=0.50:0.95, all areas, maxDets=100"),
    ("mAP_50", "AP @ IoU=0.50"),
    ("mAP_75", "AP @ IoU=0.75"),
    ("mAP_small", "AP, area < 32^2"),
    ("mAP_medium", "AP, 32^2 <= area < 96^2"),
    ("mAP_large", "AP, area >= 96^2"),
    ("AR_1", "AR, maxDets=1"),
    ("AR_10", "AR, maxDets=10"),
    ("AR_100", "AR, maxDets=100"),
    ("AR_small", "AR, area < 32^2"),
    ("AR_medium", "AR, 32^2 <= area < 96^2"),
    ("AR_large", "AR, area >= 96^2"),
]


def evaluate_coco(
    gt_json: str | Path,
    detections: list[dict],
    cat_ids: list[int] | None = None,
    image_ids: list[int] | None = None,
    per_class: bool = True,
) -> dict:
    """Run COCOeval and return the twelve summary stats, plus per-class AP.

    ``cat_ids`` restricts the evaluation to categories both sides predict.
    ``image_ids`` restricts it to the frames actually run, which matters when
    evaluating a subsample: without it, every unvisited image counts as a frame
    where the detector found nothing.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))

    if not detections:
        return {"error": "no detections", "n_detections": 0}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(detections, fh)
        det_path = fh.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(det_path)
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            if cat_ids is not None:
                ev.params.catIds = list(cat_ids)
            if image_ids is not None:
                ev.params.imgIds = list(image_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
    finally:
        Path(det_path).unlink(missing_ok=True)

    out = {name: round(float(ev.stats[i]), 4) for i, (name, _) in enumerate(COCO_STATS)}
    out["n_detections"] = len(detections)

    if per_class and cat_ids:
        names = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        # precision is [T, R, K, A, M]: iou thr, recall, category, area, maxDets.
        # Index -1 on area/maxDets picks "all areas, maxDets=100" to match mAP.
        precision = ev.eval["precision"]
        by_class = {}
        for k, cid in enumerate(ev.params.catIds):
            p = precision[:, :, k, 0, -1]
            p = p[p > -1]
            by_class[names.get(cid, str(cid))] = round(float(p.mean()), 4) if p.size else None
        out["per_class_AP"] = by_class
    return out


def summarise(label: str, stats: dict) -> str:
    """One readable line per model."""
    if "error" in stats:
        return f"{label:<28} {stats['error']}"
    return (
        f"{label:<28} mAP={stats['mAP']:.4f}  AP50={stats['mAP_50']:.4f}  "
        f"AP75={stats['mAP_75']:.4f}  AR100={stats['AR_100']:.4f}  "
        f"(small={stats['mAP_small']:.4f} medium={stats['mAP_medium']:.4f})"
    )
