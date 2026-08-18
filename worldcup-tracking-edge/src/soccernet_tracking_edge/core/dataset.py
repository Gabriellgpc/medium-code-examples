"""SoccerNet-Tracking ground truth -> a detection training set (COCO format).

The tracking annotations carry no class column: ``gt.txt`` rows are
``frame,id,x,y,w,h,1,-1,-1,-1`` for every object alike. The class lives in
``gameinfo.ini``, which maps each tracklet id to a role string::

    trackletID_4= referee;main
    trackletID_20= ball;1
    trackletID_1= player team left;4

So the label comes from joining the two files on the tracklet id. That join also
hands us the **team** and the **jersey number** for free, which later feeds team
assignment and re-identification.

Class ids follow the public fine-tuned SoccerNet head (ball=0, player=1,
referee=2, goalkeeper=3) so a model trained here is a drop-in replacement in the
existing inference path via ``config.SOCCER_TO_LOCAL``.
"""

from __future__ import annotations

import configparser
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from soccernet_tracking_edge.core import mot

BALL, PLAYER, REFEREE, GOALKEEPER = 0, 1, 2, 3
CATEGORIES = [
    {"id": BALL, "name": "ball"},
    {"id": PLAYER, "name": "player"},
    {"id": REFEREE, "name": "referee"},
    {"id": GOALKEEPER, "name": "goalkeeper"},
]


@dataclass(frozen=True)
class Tracklet:
    """One annotated object in a sequence, as described by ``gameinfo.ini``."""

    category_id: int
    team: str | None      # "left" / "right" / None
    jersey: str | None    # kept as text: values include "A", "X", "main"
    raw: str


def canonical_role(raw: str) -> int | None:
    """Map a ``gameinfo.ini`` role string to a category id.

    Deliberately prefix-based rather than exact-match: the released files are
    inconsistent (``goalkeeper team right`` in one sequence, ``goalkeepers team
    left`` in another). An exact-match table drops goalkeepers silently, which is
    the kind of bug that only shows up as a mysteriously weak class later.
    """
    r = raw.strip().lower()
    if r.startswith("ball"):
        return BALL
    if r.startswith("goalkeeper"):  # also matches the plural typo
        return GOALKEEPER
    if r.startswith("referee"):
        return REFEREE
    if r.startswith("player"):
        return PLAYER
    return None


def parse_gameinfo(path: Path) -> dict[int, Tracklet]:
    """Read ``gameinfo.ini`` into ``{tracklet_id: Tracklet}``."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # keep key case
    cfg.read(path)
    out: dict[int, Tracklet] = {}
    for key, value in cfg["Sequence"].items():
        if not key.lower().startswith("trackletid_"):
            continue
        tid = int(key.split("_", 1)[1])
        role_part, _, jersey = value.partition(";")
        cid = canonical_role(role_part)
        if cid is None:
            logger.warning(f"{path.parent.name}: unrecognised role {value!r} (tracklet {tid})")
            continue
        low = role_part.lower()
        team = "left" if "left" in low else "right" if "right" in low else None
        out[tid] = Tracklet(cid, team, jersey.strip() or None, value.strip())
    return out


def sequence_size(seq_dir: Path) -> tuple[int, int]:
    """(width, height) from ``seqinfo.ini``."""
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    return int(cfg["Sequence"]["imWidth"]), int(cfg["Sequence"]["imHeight"])


def build_coco(seq_dirs: Iterable[Path], out_json: Path, root: Path) -> dict:
    """Write a COCO detection file covering ``seq_dirs``; return per-class stats.

    ``file_name`` is stored relative to ``root`` so the frames stay where they
    are: a training set of tens of thousands of frames should not be copied.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    stats = {c["name"]: 0 for c in CATEGORIES}
    ball_sizes: list[float] = []
    orphans = 0
    img_id = ann_id = 1

    for seq_dir in sorted(seq_dirs):
        tracklets = parse_gameinfo(seq_dir / "gameinfo.ini")
        gt = mot.read_mot(seq_dir / "gt" / "gt.txt")
        width, height = sequence_size(seq_dir)
        frames = mot.list_frames(seq_dir)

        for frame_no, frame_path in enumerate(frames, start=1):
            images.append({
                "id": img_id,
                "file_name": str(frame_path.relative_to(root)),
                "width": width,
                "height": height,
                "sequence": seq_dir.name,
                "frame": frame_no,
            })
            for row in gt.get(frame_no, []):
                tid, x, y, w, h = (float(v) for v in row)
                trk = tracklets.get(int(tid))
                if trk is None:
                    orphans += 1
                    continue
                # Clip to the image: a few GT boxes run past the border.
                x0, y0 = max(0.0, x), max(0.0, y)
                x1, y1 = min(float(width), x + w), min(float(height), y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                bw, bh = x1 - x0, y1 - y0
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": trk.category_id,
                    "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                    "track_id": int(tid),
                    "team": trk.team,
                    "jersey": trk.jersey,
                })
                stats[CATEGORIES[trk.category_id]["name"]] += 1
                if trk.category_id == BALL:
                    ball_sizes.append(max(bw, bh))
                ann_id += 1
            img_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "info": {"description": "SoccerNet-Tracking as detection (ball/player/referee/goalkeeper)"},
        "images": images, "annotations": annotations, "categories": CATEGORIES,
    }))

    if orphans:
        logger.warning(f"{orphans} boxes had a tracklet id absent from gameinfo.ini")
    return {
        "images": len(images),
        "boxes": len(annotations),
        "per_class": stats,
        "ball_sizes_px": ball_sizes,
    }
