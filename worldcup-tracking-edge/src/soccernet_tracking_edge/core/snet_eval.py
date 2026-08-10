"""Evaluation, one protocol per head — deliberately not one aggregate number.

TRAINING-DESIGN section 5 is emphatic that the three heads need three protocols and
that mixing them misleads. In particular GS-HOTA does not score the ball at all,
so the ball head has to carry the WASB protocol or it has no number.

WASB's protocol: for each frame, classify the prediction against ground truth
using a distance threshold ``tau`` in pixels, then report F1, accuracy and
average precision. Note what a *true negative* means here — the model correctly
said the ball is not in frame — which is why accuracy alone flatters any method
on a dataset where the ball is visible 94% of the time.
"""

from __future__ import annotations

import numpy as np


def classify(
    pred: tuple[float, float] | None, truth: tuple[float, float] | None, tau: float
) -> str:
    """WASB's four-way outcome for one frame."""
    if truth is None:
        return "TN" if pred is None else "FP"
    if pred is None:
        return "FN"
    d = float(np.hypot(pred[0] - truth[0], pred[1] - truth[1]))
    # A prediction beyond tau is both a miss and a false alarm in WASB's counting:
    # the ball was there and we did not find it, and we claimed it somewhere else.
    return "TP" if d <= tau else "FP"


def ball_metrics(records: list[dict], tau: float = 4.0) -> dict:
    """F1 / accuracy / AP at a fixed pixel tolerance.

    ``records`` is a list of ``{"pred": (x, y) | None, "score": float,
    "truth": (x, y) | None}`` in native pixels.
    """
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in records:
        counts[classify(r["pred"], r["truth"], tau)] += 1

    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, sum(counts.values()))

    # AP over the positive predictions, ranked by confidence.
    scored = sorted(
        [r for r in records if r["pred"] is not None],
        key=lambda r: -r.get("score", 0.0),
    )
    n_truth = sum(1 for r in records if r["truth"] is not None)
    tps = fps = 0
    ap, prev_recall = 0.0, 0.0
    for r in scored:
        if classify(r["pred"], r["truth"], tau) == "TP":
            tps += 1
        else:
            fps += 1
        rec = tps / n_truth if n_truth else 0.0
        prec = tps / (tps + fps)
        ap += prec * (rec - prev_recall)
        prev_recall = rec

    return {
        "tau_px": tau, **counts,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round(accuracy, 4), "ap": round(ap, 4),
        "n_frames": len(records),
    }


def ball_metrics_sweep(records: list[dict], taus=(1, 2, 3, 4, 6, 8, 12)) -> dict:
    """WASB Fig. 6's shape: the same records at several tolerances.

    A single tau hides whether a method is precisely right or merely nearby, which
    is exactly the distinction that matters for a 15 px ball.
    """
    return {str(t): ball_metrics(records, float(t)) for t in taus}


def keypoint_metrics(errors_px: np.ndarray, budget_px: float = 3.0) -> dict:
    """Localisation error for the pitch head, against the section 6.6 budget.

    ``budget_px`` defaults to 3 because that is where 8 keypoints kept 97.9% of
    players inside GS-HOTA's 5 m tolerance. The fraction under budget is the
    number that actually predicts downstream behaviour; the median alone does not.
    """
    e = np.asarray(errors_px, dtype=np.float64)
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "median_px": round(float(np.median(e)), 3),
        "p90_px": round(float(np.percentile(e, 90)), 3),
        "pct_under_budget": round(100.0 * float((e <= budget_px).mean()), 2),
        "budget_px": budget_px,
    }
