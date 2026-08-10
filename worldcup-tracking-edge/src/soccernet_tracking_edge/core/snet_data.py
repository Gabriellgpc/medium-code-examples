"""Dataset feeding SNet: a 3-frame stack in, three heads' targets out.

Frames come from the products ``prepare_gsr.py`` already writes, so nothing here
re-parses SN-GSR: ``detection.json`` for boxes and athlete pitch coordinates,
``ball_track.json`` for the ordered per-frame ball centre.

**Where the pitch keypoint supervision comes from.** SN-GSR annotates pitch
*lines*, not landmarks, and turning polylines into named keypoints means writing
line-line and line-conic intersection code. There is a shorter path that reuses
machinery already validated: every athlete carries both an image box and a pitch
position in metres, so a homography can be fitted per frame directly, and section
6.6 measured that fit at a median residual of 0.09 m. Inverting it and projecting
the known ``LANDMARKS`` gives every landmark's true image position — including
ones whose line is occluded, which line-intersection would miss. Frames with too
few athletes to fit a trustworthy homography are simply not supervised for pitch.

Six of SN-GSR's 26 line types are goal posts and crossbars. Those are **above the
ground plane**, so they can never take part in a planar homography; they are not
used here and must not be added to the landmark set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from soccernet_tracking_edge.core.pitch import LANDMARKS, fit_homography, project
from soccernet_tracking_edge.core.targets import (
    ball_gaussian,
    detection_targets,
    keypoint_targets,
)

# detection.json class ids -> contiguous detection-head classes (ball is excluded:
# it has its own full-resolution head, which also removes the worst imbalance).
DET_CLASSES = {1: 0, 2: 1, 3: 2}   # player, referee, goalkeeper
N_DET_CLASSES = 3

MIN_ATHLETES_FOR_HOMOGRAPHY = 8


def flip_permutation() -> np.ndarray:
    """Index map for a horizontal flip of the landmark set.

    A flipped frame turns the left goal into the right goal. Feeding it with
    unpermuted labels teaches the pitch head that ``corner_tl`` is wherever the
    augmentation happened to put it — the same bug as flipping left/right joints
    in pose estimation without swapping their labels.

    Built by mirroring each landmark's pitch coordinate (x -> -x) and finding the
    landmark that actually sits there, so it stays correct if LANDMARKS changes.
    """
    names = list(LANDMARKS)
    coords = {name: (round(x, 3), round(y, 3)) for name, (x, y) in LANDMARKS.items()}
    lookup = {v: k for k, v in coords.items()}
    perm = []
    for name in names:
        x, y = coords[name]
        mirrored = lookup.get((round(-x, 3), y))
        if mirrored is None:
            raise ValueError(f"no mirror landmark for {name}; the set is not symmetric")
        perm.append(names.index(mirrored))
    return np.asarray(perm, dtype=np.int64)


class SNetDataset(Dataset):
    """One sample = frames (t-2, t-1, t) stacked on channels, plus per-head targets."""

    def __init__(
        self,
        root: str | Path,
        det_json: str | Path,
        ball_json: str | Path | None = None,
        size: tuple[int, int] = (384, 640),
        in_frames: int = 3,
        ball_stride: int = 1,
        det_stride: int = 4,
        heads: tuple[str, ...] = ("ball", "detection", "pitch"),
        augment: bool = False,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.size = size
        self.in_frames = in_frames
        self.ball_stride = ball_stride
        self.det_stride = det_stride
        self.heads = heads
        self.augment = augment
        self.flip_perm = flip_permutation()

        data = json.loads(Path(det_json).read_text())
        self.anns: dict[int, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            self.anns[a["image_id"]].append(a)

        # Group by sequence and order by file name: the temporal window depends on
        # adjacency, and adjacency is only defined inside a sequence.
        by_seq: dict[str, list[dict]] = defaultdict(list)
        for im in data["images"]:
            by_seq[im.get("sequence", "_")].append(im)
        self.sequences = {
            s: sorted(v, key=lambda i: i["file_name"]) for s, v in by_seq.items()
        }

        self.ball: dict[str, dict[str, dict]] = {}
        if ball_json is not None:
            bt = json.loads(Path(ball_json).read_text())
            for seq in bt["sequences"]:
                self.ball[seq["sequence"]] = {f["file_name"]: f for f in seq["frames"]}

        self.index: list[tuple[str, int]] = [
            (s, i) for s, frames in self.sequences.items() for i in range(len(frames))
        ]
        if limit:
            self.index = self.index[:limit]

        self.landmark_pts = np.array(list(LANDMARKS.values()), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.index)

    def _load_stack(self, seq: str, i: int, flip: bool) -> tuple[np.ndarray, float, float]:
        frames = self.sequences[seq]
        h, w = self.size
        chans = []
        for k in range(self.in_frames - 1, -1, -1):
            # Clamp at the sequence start: repeating frame 0 is honest padding.
            # Wrapping to the previous sequence would invent motion that never
            # happened, and the ball head would learn it.
            rec = frames[max(0, i - k)]
            img = cv2.imread(str(self.root / rec["file_name"]))
            if img is None:
                img = np.zeros((rec["height"], rec["width"], 3), dtype=np.uint8)
            src_h, src_w = img.shape[:2]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if flip:
                img = cv2.flip(img, 1)
            chans.append(img)
        stack = np.concatenate(chans, axis=2).astype(np.float32) / 255.0
        return stack.transpose(2, 0, 1), w / src_w, h / src_h

    def __getitem__(self, idx: int) -> dict:
        seq, i = self.index[idx]
        rec = self.sequences[seq][i]
        flip = bool(self.augment and np.random.rand() < 0.5)
        stack, sx, sy = self._load_stack(seq, i, flip)
        h, w = self.size
        out: dict[str, np.ndarray] = {"image": stack}

        anns = self.anns.get(rec["id"], [])

        if "ball" in self.heads:
            bh, bw = h // self.ball_stride, w // self.ball_stride
            heat = np.zeros((1, bh, bw), dtype=np.float32)
            xy = self._ball_xy(seq, rec, anns)
            visible = 0.0
            if xy is not None:
                cx, cy = xy[0] * sx, xy[1] * sy
                if flip:
                    cx = w - 1 - cx
                cx, cy = cx / self.ball_stride, cy / self.ball_stride
                if 0 <= cx < bw and 0 <= cy < bh:
                    ball_gaussian(heat[0], cx, cy)
                    visible = 1.0
            out["ball_heat"] = heat
            out["ball_visible"] = np.float32(visible)

        if "detection" in self.heads:
            boxes, classes = [], []
            for a in anns:
                cls = DET_CLASSES.get(a["category_id"])
                if cls is None:
                    continue
                bx, by, bwid, bhei = a["bbox"]
                bx, bwid = bx * sx, bwid * sx
                by, bhei = by * sy, bhei * sy
                if flip:
                    bx = w - bx - bwid
                boxes.append([bx, by, bwid, bhei])
                classes.append(cls)
            t = detection_targets(
                np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                np.asarray(classes, dtype=np.int64),
                h // self.det_stride, w // self.det_stride,
                N_DET_CLASSES, 1.0 / self.det_stride,
            )
            out["det_heat"] = t["heat"]
            out["det_size"] = t["size"]
            out["det_offset"] = t["offset"]
            out["det_mask"] = t["mask"]

        if "pitch" in self.heads:
            kp, present, valid = self._pitch_targets(anns, sx, sy, flip)
            out["kp_heat"] = kp
            out["kp_present"] = present
            # Per-frame: was a homography available at all? Distinct from a
            # landmark simply being out of frame (see keypoint_loss).
            out["kp_valid"] = np.float32(valid)

        return out

    def _ball_xy(self, seq: str, rec: dict, anns: list[dict]) -> tuple[float, float] | None:
        """Ball centre in native pixels, or None when it is not in frame.

        Prefers ``ball_track.json``, which is ordered and carries an explicit
        visibility flag. Falls back to the ball box in the detection annotations,
        so a dataset without the ball-track export (SoccerNet-Tracking, for
        instance) still trains the ball head.
        """
        frame = self.ball.get(seq, {}).get(rec["file_name"])
        if frame is not None:
            if frame.get("visible") and frame.get("xy"):
                return float(frame["xy"][0]), float(frame["xy"][1])
            return None
        for a in anns:
            if a["category_id"] == 0:
                bx, by, bw, bh = a["bbox"]
                return bx + bw / 2.0, by + bh / 2.0
        return None

    def _pitch_targets(
        self, anns: list[dict], sx: float, sy: float, flip: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        h, w = self.size
        n_kp = len(self.landmark_pts)
        heat = np.zeros((n_kp, h // self.ball_stride, w // self.ball_stride), dtype=np.float32)
        present = np.zeros(n_kp, dtype=np.float32)

        feet, pitch = [], []
        for a in anns:
            if a["category_id"] == 0 or not a.get("pitch_xy"):
                continue
            if a["pitch_xy"][0] is None or a["pitch_xy"][1] is None:
                continue
            bx, by, bwid, bhei = a["bbox"]
            feet.append([(bx + bwid / 2.0), by + bhei])
            pitch.append(a["pitch_xy"])
        if len(feet) < MIN_ATHLETES_FOR_HOMOGRAPHY:
            return heat, present, 0.0   # no ground truth: excluded from the loss

        try:
            # Threshold is in metres: the destination of this homography is the pitch.
            h_img2pitch, _ = fit_homography(np.asarray(feet), np.asarray(pitch), ransac_m=1.0)
            h_pitch2img = np.linalg.inv(h_img2pitch)
        except (ValueError, np.linalg.LinAlgError):
            return heat, present, 0.0

        img_pts = project(h_pitch2img, self.landmark_pts)
        img_pts[:, 0] *= sx
        img_pts[:, 1] *= sy
        if flip:
            img_pts[:, 0] = w - 1 - img_pts[:, 0]
            img_pts = img_pts[self.flip_perm]

        visible = (
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w)
            & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        )
        heat, present = keypoint_targets(
            img_pts, visible, h // self.ball_stride, w // self.ball_stride,
            n_kp, 1.0 / self.ball_stride,
        )
        return heat, present, 1.0
