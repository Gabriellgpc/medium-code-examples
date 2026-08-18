"""Project constants and artifact paths.

Paths are resolved from the nearest ``pyproject.toml`` (so commands work from any
CWD) and are env-overridable, which is how we keep every download and cache on
the big workspace disk instead of a near-full root partition::

    export WCT_DATA_DIR=/media/.../data
    export HF_HOME=/media/.../data/hf   # keep HF model cache off root too
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Model / preprocessing ------------------------------------------------
RFDETR_RESOLUTION = 384  # square export size; the iGPU wants a static shape
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS = ("rfdetr",)
PRECISIONS = ("fp32", "fp16", "int8_woq", "int8_full")
TRACKERS = ("bytetrack", "ocsort")

# RF-DETR's pretrained head emits raw COCO-91 class ids (91 slots, with gaps).
# For football we only care about people on the pitch and the ball. In COCO-91,
# person = 1 and "sports ball" = 37. We keep these two and drop everything else.
COCO91_PERSON = 1
COCO91_SPORTS_BALL = 37
FOOTBALL_KEEP_IDS = (COCO91_PERSON, COCO91_SPORTS_BALL)

# Our own compact label space used downstream (detector + MOT files).
CLASS_PERSON = 0
CLASS_BALL = 1
CLASS_NAMES = {CLASS_PERSON: "person", CLASS_BALL: "ball"}
COCO91_TO_LOCAL = {COCO91_PERSON: CLASS_PERSON, COCO91_SPORTS_BALL: CLASS_BALL}

# Fine-tuned SoccerNet checkpoint (julianzu9612/RFDETR-Soccernet, Apache-2.0).
# Verified empirically: the exported head has width exactly 4 (no N+1 dummy slot),
# so classes are 0-based ball=0, player=1, referee=2, goalkeeper=3 — matching
# config.json and the PyTorch predict path. We collapse the three people roles
# into our single "person" class and keep the ball, so tracking is evaluated the
# same way as the COCO path.
SOCCER_HF_REPO = "julianzu9612/RFDETR-Soccernet"
SOCCER_TO_LOCAL = {  # exported (0-based) head id -> local class
    0: CLASS_BALL,      # ball
    1: CLASS_PERSON,    # player
    2: CLASS_PERSON,    # referee
    3: CLASS_PERSON,    # goalkeeper
}


def find_project_root() -> Path:
    """Return the project root (nearest ancestor with a ``pyproject.toml``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


PROJECT_ROOT = find_project_root()
DATA_DIR = _dir("WCT_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _dir("WCT_MODELS_DIR", PROJECT_ROOT / "models")
OUTPUT_DIR = _dir("WCT_OUTPUT_DIR", PROJECT_ROOT / "output")

# SoccerNet-Tracking layout (MOTChallenge): one sequence dir with img1/ + gt/gt.txt.
SOCCERNET_DIR = DATA_DIR / "soccernet"
DEFAULT_SEQUENCE = "SNMOT-116"


def ir_path(precision: str, tag: str = "det") -> Path:
    """Canonical IR path for a (tag, precision), the single source of truth.

    ``tag`` selects the detector head: ``"det"`` = pretrained-COCO RF-DETR-Nano
    (person + sports ball), ``"soccer"`` = the fine-tuned SoccerNet checkpoint
    (dedicated ball class). Both share the export/quantize/track pipeline.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    return MODELS_DIR / f"rfdetr_{tag}_{precision}.xml"
