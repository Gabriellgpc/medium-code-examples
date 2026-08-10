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
        kp_stride: int = 4,
        heads: tuple[str, ...] = ("ball", "detection", "pitch"),
        augment: bool = False,
        limit: int | None = None,
        stride: int | None = None,
    ):
        self.root = Path(root)
        self.size = size
        self.in_frames = in_frames
        self.ball_stride = ball_stride
        self.det_stride = det_stride
        self.kp_stride = kp_stride
        self.heads = heads
        self.augment = augment
        self.flip_perm = flip_permutation()
        self.augmenter = self._build_augmenter() if augment else None

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
        if stride and stride > 1:
            # Subsample *across* sequences. Taking a prefix instead, as `limit`
            # does, silently evaluates on a single 750-frame clip: the first
            # validation run reported TN=0 because that one clip happens to have
            # the ball visible in every frame, so the metric never tested whether
            # the head can say "not in frame" at all.
            self.index = self.index[::stride]
        if limit:
            self.index = self.index[:limit]

        self.landmark_pts = np.array(list(LANDMARKS.values()), dtype=np.float64)

    def _build_augmenter(self):
        """Photometric and mild geometric augmentation, identical across the stack.

        ``additional_targets`` is what makes the three frames share one draw of
        the parameters. If each frame sampled its own shift or rotation, the
        difference between consecutive frames would be augmentation rather than
        the ball moving, and the temporal head would learn that.

        Horizontal flip is deliberately NOT here. Albumentations mirrors keypoint
        *coordinates*, but it has no idea that ``corner_tl`` becomes ``corner_tr``
        — that relabelling is semantic and stays ours (see ``flip_permutation``).
        Geometry stays gentle for the same reason the ball head exists at all: at
        384x640 the ball is about 5 px across, so aggressive scaling or blur
        deletes the object we are trying to find.
        """
        import albumentations as A

        return A.Compose(
            [
                A.Affine(scale=(0.92, 1.08), translate_percent=(-0.04, 0.04),
                         rotate=(-4, 4), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20,
                                     val_shift_limit=12, p=0.3),
                A.GaussNoise(p=0.15),
                A.MotionBlur(blur_limit=3, p=0.08),
            ],
            additional_targets={"image1": "image", "image2": "image"},
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"],
                                     clip=True, min_visibility=0.25),
            # remove_invisible=False is load-bearing: keypoint index k IS landmark
            # k, and silently dropping the ones that leave frame would shift every
            # later landmark into the wrong heatmap channel.
            keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_ids"],
                                            remove_invisible=False),
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_frames(self, seq: str, i: int, flip: bool) -> tuple[list[np.ndarray], float, float]:
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
        return chans, w / src_w, h / src_h

    def __getitem__(self, idx: int) -> dict:
        seq, i = self.index[idx]
        rec = self.sequences[seq][i]
        h, w = self.size
        flip = bool(self.augment and np.random.rand() < 0.5)
        frames, sx, sy = self._load_frames(seq, i, flip)
        anns = self.anns.get(rec["id"], [])

        # Everything is converted to target-resolution pixels first, then
        # augmented as coordinates, and only rasterised into heatmaps at the end.
        # Rasterising first and warping the heatmap would blur the Gaussians and
        # move their peaks off the true centre.
        boxes: list[list[float]] = []
        classes: list[int] = []
        if "detection" in self.heads:
            for a in anns:
                cls = DET_CLASSES.get(a["category_id"])
                if cls is None:
                    continue
                bx, by, bwid, bhei = a["bbox"]
                bx, bwid = bx * sx, bwid * sx
                by, bhei = by * sy, bhei * sy
                if flip:
                    bx = w - bx - bwid
                bx = max(0.0, min(bx, w - 1.0))
                by = max(0.0, min(by, h - 1.0))
                bwid = max(1e-3, min(bwid, w - bx))
                bhei = max(1e-3, min(bhei, h - by))
                boxes.append([bx, by, bwid, bhei])
                classes.append(cls)

        ball_xy = None
        if "ball" in self.heads:
            xy = self._ball_xy(seq, rec, anns)
            if xy is not None:
                cx, cy = xy[0] * sx, xy[1] * sy
                if flip:
                    cx = w - 1 - cx
                ball_xy = [cx, cy]

        landmark_xy, kp_valid = (None, 0.0)
        if "pitch" in self.heads:
            landmark_xy, kp_valid = self._landmark_points(anns, sx, sy, flip)

        # One keypoint list: id 0 is the ball, ids 1..N are landmarks. They travel
        # together so a single transform moves both consistently.
        kp_xy: list[list[float]] = []
        kp_ids: list[int] = []
        if ball_xy is not None:
            kp_xy.append(ball_xy)
            kp_ids.append(0)
        if landmark_xy is not None:
            for k, (x, y) in enumerate(landmark_xy):
                kp_xy.append([float(x), float(y)])
                kp_ids.append(1 + k)

        if self.augmenter is not None:
            res = self.augmenter(
                image=frames[0], image1=frames[1], image2=frames[2],
                bboxes=boxes, bbox_classes=classes,
                keypoints=kp_xy, kp_ids=kp_ids,
            )
            frames = [res["image"], res["image1"], res["image2"]]
            boxes = [list(b) for b in res["bboxes"]]
            classes = [int(round(float(c))) for c in res["bbox_classes"]]
            kp_xy = [list(k) for k in res["keypoints"]]
            # Albumentations returns label fields as floats; the ids index
            # heatmap channels, so they go back to int before use.
            kp_ids = [int(round(float(k))) for k in res["kp_ids"]]

        by_id = dict(zip(kp_ids, kp_xy, strict=True))
        stack = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        out: dict[str, np.ndarray] = {"image": stack.transpose(2, 0, 1)}

        if "ball" in self.heads:
            bh, bw = h // self.ball_stride, w // self.ball_stride
            heat = np.zeros((1, bh, bw), dtype=np.float32)
            visible = 0.0
            pt = by_id.get(0)
            if pt is not None:
                cx, cy = pt[0] / self.ball_stride, pt[1] / self.ball_stride
                if 0 <= cx < bw and 0 <= cy < bh:
                    ball_gaussian(heat[0], cx, cy)
                    visible = 1.0
            out["ball_heat"] = heat
            out["ball_visible"] = np.float32(visible)

        if "detection" in self.heads:
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
            n_kp = len(self.landmark_pts)
            pts = np.zeros((n_kp, 2), dtype=np.float64)
            vis = np.zeros(n_kp, dtype=bool)
            for k in range(n_kp):
                pt = by_id.get(1 + k)
                if pt is None:
                    continue
                pts[k] = pt
                vis[k] = 0 <= pt[0] < w and 0 <= pt[1] < h
            # Rasterised at kp_stride, not at full resolution: 33 channels of
            # 384x640 is 32 MB per sample and 260 MB per batch, which starves the
            # loader. Trunk stride costs 2.0 MB and decodes to a median 0.66
            # native px, well inside the section 6.6 budget.
            heat, present = keypoint_targets(
                pts, vis, h // self.kp_stride, w // self.kp_stride,
                n_kp, 1.0 / self.kp_stride, sigma=2.0,
            )
            out["kp_heat"] = heat
            out["kp_present"] = present
            # Per-frame: was a homography available at all? Distinct from a
            # landmark simply being out of frame (see keypoint_loss).
            out["kp_valid"] = np.float32(kp_valid)

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

    def _landmark_points(
        self, anns: list[dict], sx: float, sy: float, flip: bool
    ) -> tuple[np.ndarray | None, float]:
        """Landmark positions in target-resolution pixels, or None when unsupervised.

        Returns points rather than heatmaps so they can pass through augmentation
        as coordinates; the caller rasterises afterwards.
        """
        w = self.size[1]
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
            return None, 0.0   # no ground truth: excluded from the loss

        try:
            # Threshold is in metres: the destination of this homography is the pitch.
            h_img2pitch, _ = fit_homography(np.asarray(feet), np.asarray(pitch), ransac_m=1.0)
            h_pitch2img = np.linalg.inv(h_img2pitch)
        except (ValueError, np.linalg.LinAlgError):
            return None, 0.0

        img_pts = project(h_pitch2img, self.landmark_pts)
        img_pts[:, 0] *= sx
        img_pts[:, 1] *= sy
        if flip:
            img_pts[:, 0] = w - 1 - img_pts[:, 0]
            img_pts = img_pts[self.flip_perm]
        return img_pts, 1.0
