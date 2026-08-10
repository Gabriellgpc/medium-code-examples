"""Score a MOT prediction file against sequence ground truth.

Reads MOTChallenge GT (``<seq>/gt/gt.txt``) and a prediction file produced by
``snt-track``, then prints MOTA / MOTP / IDF1 / ID-switches via ``motmetrics``.
Optionally cross-checks HOTA with ``sn-trackeval`` when installed (``--extra hota``).
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger
from tabulate import tabulate

from soccernet_tracking_edge.core import mot


@click.command()
@click.option("--sequence", type=click.Path(exists=True, path_type=Path), required=True,
              help="SoccerNet sequence dir containing gt/gt.txt.")
@click.option("--pred", type=click.Path(exists=True, path_type=Path), required=True,
              help="Prediction file (MOTChallenge format) from snt-track.")
def evaluate(sequence: Path, pred: Path) -> None:
    """Compute MOT metrics for a prediction file vs ground truth."""
    gt_path = sequence / "gt" / "gt.txt"
    gt = mot.read_mot(gt_path)
    pred_boxes = mot.read_mot(pred)
    logger.info(f"GT frames: {len(gt)} · pred frames: {len(pred_boxes)}")

    metrics = mot.evaluate(gt, pred_boxes)
    rows = [[k, v] for k, v in metrics.items()]
    click.echo(tabulate(rows, headers=["metric", "value"], tablefmt="github"))
