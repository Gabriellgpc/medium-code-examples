"""Drawing helpers: tracked boxes with stable per-ID colors, labels, and FPS."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from soccernet_tracking_edge.config import CLASS_BALL, CLASS_NAMES


def _annotators() -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    # Color by tracker_id so an identity keeps its color across frames; an ID
    # switch is visible as a box abruptly changing color.
    box = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK, text_scale=0.4)
    return box, label


def annotate(frame: np.ndarray, det: sv.Detections, fps: float | None = None) -> np.ndarray:
    """Draw tracked detections (id + class) and an optional FPS badge."""
    out = frame.copy()
    box, label = _annotators()
    if det.tracker_id is not None and len(det):
        labels = [
            f"#{tid} {CLASS_NAMES.get(int(cid), '?')}"
            for tid, cid in zip(det.tracker_id, det.class_id, strict=False)
        ]
        out = box.annotate(out, det)
        out = label.annotate(out, det, labels=labels)
    if fps is not None:
        cv2.putText(
            out, f"{fps:5.1f} FPS", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
    return out


def ball_count(det: sv.Detections) -> int:
    """How many ball detections survived — a quick sanity readout for the ball."""
    if det.class_id is None:
        return 0
    return int((det.class_id == CLASS_BALL).sum())
