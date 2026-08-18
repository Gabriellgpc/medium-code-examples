"""RF-DETR preprocessing, sigmoid decode, and an OpenVINO detector wrapper.

RF-DETR is NMS-free: the head emits a fixed set of 300 queries, each a box
(cxcywh, normalized) plus per-class logits. We take a **sigmoid** over the
logits (focal-style head — not softmax), keep the top class per query, threshold,
and map RF-DETR's COCO-91 ids to our compact person/ball label space.

The same numpy ``preprocess`` is used for inference *and* NNCF calibration, so
the quantized activation statistics match the real input distribution.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from soccernet_tracking_edge.config import (
    COCO91_TO_LOCAL,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from soccernet_tracking_edge.core.common import OVDetModel

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def preprocess(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 image -> (1,3,size,size) float32, RF-DETR normalization.

    RF-DETR squashes to a square (no letterbox); we undo that squash in decode by
    scaling boxes back to the original width/height, so aspect ratio is handled
    at the box level rather than with padding.
    """
    resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    chw = (chw - _MEAN) / _STD
    return chw[None, ...].astype(np.float32)


def decode(
    outputs: list[np.ndarray],
    orig_hw: tuple[int, int],
    threshold: float,
    mapping: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RF-DETR outputs -> (boxes_xyxy_pixels, scores, local_class_ids).

    ``outputs[0]`` = boxes (300,4) cxcywh normalized; ``outputs[1]`` = logits
    (300, C). Sigmoid, argmax, threshold, then remap raw head ids -> our local
    person/ball ids via ``mapping`` (COCO-91 for the stock head, the fine-tuned
    soccer head otherwise). Queries whose top class isn't in ``mapping`` are
    dropped. argmax over the logits width handles either head automatically.
    """
    if mapping is None:
        mapping = COCO91_TO_LOCAL
    boxes = outputs[0][0]  # (300, 4)
    logits = outputs[1][0]  # (300, C)
    scores_all = 1.0 / (1.0 + np.exp(-logits))
    top_scores = scores_all.max(axis=1)
    top_ids = scores_all.argmax(axis=1)

    keep = (top_scores > threshold) & np.isin(top_ids, list(mapping.keys()))
    if not keep.any():
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, int)

    boxes = boxes[keep]
    scores = top_scores[keep].astype(np.float32)
    raw_ids = top_ids[keep]
    local_ids = np.array([mapping.get(int(c), -1) for c in raw_ids], dtype=int)

    h, w = orig_hw
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return xyxy, scores, local_ids


class RFDETRDetector:
    """OpenVINO RF-DETR detector that returns ``supervision.Detections``.

    Detections come out ready to hand straight to a tracker's ``update()``.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "GPU",
        threshold: float = 0.4,
        accurate: bool = False,
        mapping: dict[int, int] | None = None,
    ) -> None:
        self.model = OVDetModel(model_path, device=device, accurate=accurate)
        self.threshold = threshold
        self.size = self.model.in_w
        self.mapping = mapping  # None => COCO-91 head; pass SOCCER_TO_LOCAL for soccer

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        return preprocess(img_bgr, self.size)

    def detect(self, img_bgr: np.ndarray) -> sv.Detections:
        tensor = self.preprocess(img_bgr)
        outputs = self.model.forward(tensor)
        xyxy, scores, class_ids = decode(outputs, img_bgr.shape[:2], self.threshold, self.mapping)
        if len(xyxy) == 0:
            return sv.Detections.empty()
        return sv.Detections(xyxy=xyxy, confidence=scores, class_id=class_ids)
