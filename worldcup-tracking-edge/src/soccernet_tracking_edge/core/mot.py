"""MOTChallenge-format I/O and multi-object-tracking evaluation.

SoccerNet-Tracking ships each sequence as a MOTChallenge directory::

    <seq>/img1/000001.jpg ...        # extracted frames, 6-digit names
    <seq>/gt/gt.txt                  # frame,id,x,y,w,h,conf,cls,vis  (CSV)
    <seq>/seqinfo.ini

``gt.txt`` / prediction rows are ``frame,id,bb_left,bb_top,bb_width,bb_height,
conf,-1,-1,-1``. We read/write that format so predictions drop straight into
``motmetrics`` (and, optionally, ``sn-trackeval`` for HOTA) with zero conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def list_frames(seq_dir: Path) -> list[Path]:
    """Sorted list of image frames in ``<seq>/img1``."""
    img1 = seq_dir / "img1"
    return sorted(img1.glob("*.jpg")) + sorted(img1.glob("*.png"))


def read_mot(path: Path) -> dict[int, np.ndarray]:
    """Read a MOTChallenge file into ``{frame: array(N,5)}`` of ``[id,x,y,w,h]``.

    Frames are 1-indexed (MOTChallenge convention). Rows with conf == 0 are
    ignored (some GT files flag ignore-regions that way).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.size == 0:
        return {}
    out: dict[int, list[list[float]]] = {}
    for row in raw:
        frame = int(row[0])
        conf = row[6] if len(row) > 6 else 1.0
        if conf == 0:
            continue
        out.setdefault(frame, []).append([row[1], row[2], row[3], row[4], row[5]])
    return {f: np.asarray(v, dtype=np.float32) for f, v in out.items()}


def write_mot(path: Path, rows: list[tuple[int, int, float, float, float, float]]) -> None:
    """Write ``(frame, id, x, y, w, h)`` rows in MOTChallenge format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for frame, tid, x, y, w, h in rows:
            fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def evaluate(gt: dict[int, np.ndarray], pred: dict[int, np.ndarray]) -> dict[str, float]:
    """Compute MOTA/MOTP/IDF1/ID-switches via ``motmetrics``.

    ``gt`` and ``pred`` are ``{frame: array(N,5)}`` with rows ``[id,x,y,w,h]``
    (top-left + size, MOTChallenge boxes). Matching uses IoU with a 0.5 gate.
    """
    # motmetrics 1.4 still calls np.asfarray, removed in NumPy 2.0. Restore a
    # drop-in before importing so the (otherwise fine) library works on numpy>=2.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted(set(gt) | set(pred))
    for frame in frames:
        g = gt.get(frame, np.empty((0, 5), np.float32))
        p = pred.get(frame, np.empty((0, 5), np.float32))
        gids = g[:, 0].astype(int).tolist()
        pids = p[:, 0].astype(int).tolist()
        # motmetrics wants [x,y,w,h]; iou_matrix returns distance = 1 - IoU.
        dist = mm.distances.iou_matrix(g[:, 1:5], p[:, 1:5], max_iou=0.5)
        acc.update(gids, pids, dist, frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives", "num_misses"],
        name="seq",
    )
    row = summary.loc["seq"]
    return {
        "MOTA": float(row["mota"]),
        "MOTP": float(row["motp"]),
        "IDF1": float(row["idf1"]),
        "IDSW": int(row["num_switches"]),
        "FP": int(row["num_false_positives"]),
        "FN": int(row["num_misses"]),
    }
