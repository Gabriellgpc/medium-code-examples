"""SoccerNet Game State Reconstruction (SN-GSR-2025) -> training data.

Each sequence ships one ``Labels-GameState.json`` holding everything at once::

    categories: 1 player, 2 goalkeeper, 3 referee, 4 ball, 5 pitch
    annotations[i]:
        bbox_image   {x, y, w, h}          box in pixels
        bbox_pitch   {x_bottom_middle, …}  position on the pitch, in metres
        attributes   {role, jersey, team}
        track_id                            identity through time
    annotations for category 5: {lines: {"<line name>": [{x, y}, …]}}  normalised

Two products come out of that, and they are deliberately separate files:

* a **detection** set (COCO), for the player/ball detector;
* a **pitch-line** set, for the field-registration model, which needs polylines
  rather than boxes.

Class ids are remapped to the same convention ``dataset.py`` uses (ball=0,
player=1, referee=2, goalkeeper=3), which is the public fine-tuned SoccerNet
head, so a model trained from either source is a drop-in for the existing
inference path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from soccernet_tracking_edge.core.dataset import BALL, CATEGORIES, GOALKEEPER, PLAYER, REFEREE

LABELS_FILE = "Labels-GameState.json"

# GSR category id -> our class id
GSR_TO_LOCAL = {1: PLAYER, 2: GOALKEEPER, 3: REFEREE, 4: BALL}
PITCH_CATEGORY = 5


def is_gamestate(seq_dir: Path) -> bool:
    """True when the sequence is SN-GSR (rather than SoccerNet-Tracking)."""
    return (seq_dir / LABELS_FILE).exists()


def load(seq_dir: Path) -> dict:
    return json.loads((seq_dir / LABELS_FILE).read_text())


def build_coco(seq_dirs: Iterable[Path], out_json: Path, root: Path) -> dict:
    """Write a COCO detection file; return per-class stats.

    Frames flagged ``is_labeled: false`` are skipped: keeping them would feed the
    detector images whose empty annotation list means "nobody looked", not
    "nothing there", and the model would learn to suppress real objects.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    stats = {c["name"]: 0 for c in CATEGORIES}
    ball_sizes: list[float] = []
    skipped_unlabeled = 0
    img_id = ann_id = 1

    for seq_dir in sorted(seq_dirs):
        data = load(seq_dir)
        by_image: dict[str, list[dict]] = {}
        for a in data["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        for im in data["images"]:
            if not im.get("is_labeled", True):
                skipped_unlabeled += 1
                continue
            width, height = int(im["width"]), int(im["height"])
            frame_path = seq_dir / im.get("im_dir", "img1") / im["file_name"]
            images.append({
                "id": img_id,
                "file_name": str(frame_path.relative_to(root)),
                "width": width,
                "height": height,
                "sequence": seq_dir.name,
                "gsr_image_id": im["image_id"],
            })

            for a in by_image.get(im["image_id"], []):
                cid = GSR_TO_LOCAL.get(a.get("category_id"))
                if cid is None:          # pitch lines are handled separately
                    continue
                box = a.get("bbox_image")
                if not box:
                    continue
                x0 = max(0.0, float(box["x"]))
                y0 = max(0.0, float(box["y"]))
                x1 = min(float(width), float(box["x"]) + float(box["w"]))
                y1 = min(float(height), float(box["y"]) + float(box["h"]))
                if x1 <= x0 or y1 <= y0:
                    continue
                bw, bh = x1 - x0, y1 - y0
                attrs = a.get("attributes") or {}
                pitch_xy = (a.get("bbox_pitch") or {})
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                    "track_id": a.get("track_id"),
                    "team": attrs.get("team"),
                    "jersey": attrs.get("jersey"),
                    # Ground-truth position on the minimap, in metres. This is the
                    # target the homography work is ultimately judged against.
                    "pitch_xy": [
                        pitch_xy.get("x_bottom_middle"),
                        pitch_xy.get("y_bottom_middle"),
                    ] if pitch_xy else None,
                })
                stats[CATEGORIES[cid]["name"]] += 1
                if cid == BALL:
                    ball_sizes.append(max(bw, bh))
                ann_id += 1
            img_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "info": {"description": "SN-GSR-2025 as detection (ball/player/referee/goalkeeper)"},
        "images": images, "annotations": annotations, "categories": CATEGORIES,
    }))
    if skipped_unlabeled:
        logger.info(f"skipped {skipped_unlabeled} frames flagged is_labeled=false")
    return {
        "images": len(images),
        "boxes": len(annotations),
        "per_class": stats,
        "ball_sizes_px": ball_sizes,
    }


def build_ball_track(seq_dirs: Iterable[Path], out_json: Path, root: Path) -> dict:
    """Export the ball as a per-frame centre point, in frame order.

    Box-based detection is the wrong shape for the ball: at a median of ~10 px it
    is a few pixels wide, and the heatmap trackers built for this problem
    (TrackNet, WASB) consume *consecutive frames* and regress a centre, not a
    box. So this export keeps every frame of every sequence in order, with the
    ball centre or ``null`` when it is absent, which is exactly what a temporal
    window sampler needs.

    ``visible`` is False for frames where the ball has no annotation. Those
    frames are not padding to be dropped: a tracker has to learn that the ball
    leaves the frame, and dropping them would break frame adjacency anyway.
    """
    sequences: list[dict] = []
    total = visible_total = 0

    for seq_dir in sorted(seq_dirs):
        data = load(seq_dir)
        ball_by_image: dict[str, dict] = {}
        for a in data["annotations"]:
            if a.get("category_id") == 4 and a.get("bbox_image"):
                ball_by_image[a["image_id"]] = a

        frames = []
        im_dir = next((im.get("im_dir", "img1") for im in data["images"]), "img1")
        for im in sorted(data["images"], key=lambda i: i["file_name"]):
            if not im.get("is_labeled", True):
                continue
            a = ball_by_image.get(im["image_id"])
            box = a.get("bbox_image") if a else None
            frames.append({
                "file_name": str((seq_dir / im_dir / im["file_name"]).relative_to(root)),
                "visible": box is not None,
                "xy": [round(float(box["x_center"]), 2), round(float(box["y_center"]), 2)]
                      if box else None,
                "size": round(max(float(box["w"]), float(box["h"])), 2) if box else None,
            })
            total += 1
            visible_total += box is not None

        sequences.append({
            "sequence": seq_dir.name,
            "width": data["images"][0]["width"],
            "height": data["images"][0]["height"],
            "frames": frames,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"sequences": sequences}))
    return {
        "sequences": len(sequences),
        "frames": total,
        "frames_with_ball": visible_total,
        "ball_visible_pct": round(100.0 * visible_total / total, 2) if total else 0.0,
    }


def build_pitch_lines(seq_dirs: Iterable[Path], out_json: Path, root: Path) -> dict:
    """Export the pitch-line polylines, denormalised to pixels.

    The released coordinates are normalised to [0, 1] and can fall slightly
    outside it where a line is extrapolated past the frame; they are kept as-is
    rather than clipped, because a registration model wants the true geometry,
    and clipping would bend lines toward the border.
    """
    frames: list[dict] = []
    line_names: dict[str, int] = {}

    for seq_dir in sorted(seq_dirs):
        data = load(seq_dir)
        sizes = {im["image_id"]: (int(im["width"]), int(im["height"])) for im in data["images"]}
        files = {im["image_id"]: im["file_name"] for im in data["images"]}
        im_dir = next((im.get("im_dir", "img1") for im in data["images"]), "img1")

        for a in data["annotations"]:
            if a.get("category_id") != PITCH_CATEGORY:
                continue
            gid = a["image_id"]
            if gid not in sizes:
                continue
            width, height = sizes[gid]
            lines = {}
            for name, pts in (a.get("lines") or {}).items():
                line_names[name] = line_names.get(name, 0) + 1
                lines[name] = [[round(p["x"] * width, 2), round(p["y"] * height, 2)] for p in pts]
            if not lines:
                continue
            frames.append({
                "file_name": str((seq_dir / im_dir / files[gid]).relative_to(root)),
                "sequence": seq_dir.name,
                "width": width,
                "height": height,
                "lines": lines,
            })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"frames": frames, "line_names": sorted(line_names)}))
    return {
        "frames": len(frames),
        "distinct_lines": len(line_names),
        "line_frequency": dict(sorted(line_names.items(), key=lambda kv: -kv[1])),
    }
