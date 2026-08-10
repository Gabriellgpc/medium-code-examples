"""Turn a SoccerNet-Tracking split into a COCO detection set, and report on it.

    snt-prepare --split-dir data/soccernet/tracking-2023/test --out output/coco

Sequences are split by *sequence*, never by frame: consecutive frames of one clip
are near-duplicates, so a frame-level shuffle leaks the validation set into
training and every number afterwards is optimistic.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from loguru import logger

from soccernet_tracking_edge.config import OUTPUT_DIR
from soccernet_tracking_edge.core import dataset, gamestate


def _report(name: str, stats: dict) -> None:
    logger.info(f"{name}: {stats['images']} images, {stats['boxes']} boxes")
    for cls, n in stats["per_class"].items():
        share = 100.0 * n / stats["boxes"] if stats["boxes"] else 0.0
        logger.info(f"    {cls:11s} {n:7d}  ({share:5.2f}%)")
    sizes = np.asarray(stats["ball_sizes_px"], dtype=np.float64)
    if len(sizes):
        logger.info(
            f"    ball box, longest side: median {np.median(sizes):.1f} px, "
            f"p10 {np.percentile(sizes, 10):.1f}, p90 {np.percentile(sizes, 90):.1f} "
            f"(n={len(sizes)})"
        )


@click.command("prepare")
@click.option("--split-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="A split directory holding sequence folders (SNMOT-xxx).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output directory for the COCO files (default output/coco).")
@click.option("--val-fraction", default=0.2, show_default=True,
              help="Fraction of *sequences* held out for validation.")
@click.option("--seed", default=0, show_default=True, help="Sequence-shuffle seed.")
def prepare(split_dir: Path, out: Path | None, val_fraction: float, seed: int) -> None:
    """Build train/val COCO annotations from a SoccerNet split.

    The source is detected from the sequence layout: SN-GSR ships a
    ``Labels-GameState.json`` per sequence, SoccerNet-Tracking a ``gt/gt.txt``.
    """
    gsr = sorted(p for p in split_dir.iterdir() if p.is_dir() and gamestate.is_gamestate(p))
    tracking = sorted(
        p for p in split_dir.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists()
    )
    if gsr and tracking:
        raise click.UsageError(f"{split_dir} mixes SN-GSR and Tracking sequences; split them")
    seqs, source = (gsr, "gamestate") if gsr else (tracking, "tracking")
    if not seqs:
        raise click.UsageError(
            f"no sequences under {split_dir} (expected Labels-GameState.json or gt/gt.txt)"
        )
    logger.info(f"source detected: {source}")
    builder = gamestate.build_coco if source == "gamestate" else dataset.build_coco

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_fraction))) if len(seqs) > 1 else 0
    val = [seqs[i] for i in order[:n_val]]
    train = [seqs[i] for i in order[n_val:]]
    logger.info(f"{len(seqs)} sequences → {len(train)} train, {len(val)} val (split by sequence)")

    out = out or (OUTPUT_DIR / "coco")
    root = split_dir
    summary = {}
    for name, group in (("train", train), ("val", val)):
        if not group:
            continue
        stats = builder(group, out / f"{name}.json", root)
        _report(name, stats)
        if source == "gamestate":
            lines = gamestate.build_pitch_lines(group, out / f"{name}_pitch_lines.json", root)
            logger.info(f"    pitch lines: {lines['frames']} frames, "
                        f"{lines['distinct_lines']} distinct line types")
            ball = gamestate.build_ball_track(group, out / f"{name}_ball_track.json", root)
            logger.info(f"    ball track: {ball['sequences']} sequences, {ball['frames']} frames, "
                        f"ball visible in {ball['ball_visible_pct']}%")
        summary[name] = {
            "sequences": [p.name for p in group],
            "images": stats["images"],
            "boxes": stats["boxes"],
            "per_class": stats["per_class"],
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"COCO annotations + summary → {out}")
    logger.info(f"images stay in place; file_name is relative to {root}")
