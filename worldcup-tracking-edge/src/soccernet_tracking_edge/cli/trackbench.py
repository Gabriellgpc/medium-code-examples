"""Tracking-quality sweep: {detector} × {precision} × {tracker} → MOT metrics.

Runs detect→track over one SoccerNet sequence and scores predictions against GT
with motmetrics. This is the article's propagation table: how detector choice
(Nano-COCO vs fine-tuned soccer) and quantization move MOTA / IDF1 / ID-switches.
Frames are streamed from disk (never all held in RAM) so a 128M-param detector at
1080p doesn't blow memory. Saves output/tracking.md + tracking_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger
from tabulate import tabulate

from soccernet_tracking_edge.config import OUTPUT_DIR, SOCCER_TO_LOCAL, ir_path
from soccernet_tracking_edge.core import mot
from soccernet_tracking_edge.core.rfdetr import RFDETRDetector
from soccernet_tracking_edge.core.tracking import make_tracker

# (tag, precision) detector configs. Soccer INT8 on the iGPU needs ACCURACY mode
# (PERFORMANCE corrupts the quantized DINOv2), so we load every detector accurate.
CONFIGS = [("det", "fp32"), ("det", "int8_full"), ("soccer", "fp32"), ("soccer", "int8_full")]
TRACKERS = ["bytetrack", "ocsort"]


@click.command("trackbench")
@click.option("--sequence", type=click.Path(exists=True, path_type=Path), required=True,
              help="SoccerNet sequence dir (img1/ + gt/gt.txt).")
@click.option("--device", default="GPU.0", show_default=True)
@click.option("--max-frames", default=375, show_default=True)
@click.option("--fps", default=25.0, show_default=True)
def trackbench(sequence: Path, device: str, max_frames: int, fps: float) -> None:
    """Sweep detectors × precisions × trackers → MOT metrics table."""
    gt_full = mot.read_mot(sequence / "gt" / "gt.txt")
    frame_paths = mot.list_frames(sequence)[:max_frames]
    n = len(frame_paths)
    # Restrict GT to the processed range, else later frames become phantom misses.
    gt = {f: b for f, b in gt_full.items() if f <= n}
    logger.info(f"{n} frames from {sequence.name}; {sum(len(v) for v in gt.values())} GT boxes")

    results = []
    for tag, prec in CONFIGS:
        if not ir_path(prec, tag).exists():
            logger.warning(f"skip {tag}/{prec}: IR missing")
            continue
        mapping = SOCCER_TO_LOCAL if tag == "soccer" else None
        det = RFDETRDetector(ir_path(prec, tag), device=device, threshold=0.4,
                             accurate=True, mapping=mapping)
        # Stream frames from disk; cache only the small detection objects.
        per_frame = [det.detect(cv2.imread(str(fp))) for fp in frame_paths]
        del det
        for tname in TRACKERS:
            trk = make_tracker(tname, frame_rate=fps)
            pred: dict[int, list] = {}
            for i, d in enumerate(per_frame, start=1):
                td = trk.update(d)
                if td.tracker_id is not None:
                    for box, tid in zip(td.xyxy, td.tracker_id, strict=False):
                        x1, y1, x2, y2 = box
                        pred.setdefault(i, []).append(
                            [int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        )
            m = mot.evaluate(gt, {f: np.asarray(v, np.float32) for f, v in pred.items()})
            results.append({"detector": tag, "precision": prec, "tracker": tname, **m})
            logger.info(f"{tag:6s} {prec:10s} {tname:9s}  "
                        f"MOTA={m['MOTA']:+.3f} IDF1={m['IDF1']:.3f} IDSW={m['IDSW']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tracking_results.json").write_text(json.dumps(results, indent=2))
    table = tabulate(
        [[r["detector"], r["precision"], r["tracker"], f"{r['MOTA']:+.3f}",
          f"{r['IDF1']:.3f}", r["IDSW"], r["FP"], r["FN"]] for r in results],
        headers=["detector", "precision", "tracker", "MOTA", "IDF1", "IDSW", "FP", "FN"],
        tablefmt="github",
    )
    (OUTPUT_DIR / "tracking.md").write_text(
        f"# Tracking quality — {sequence.name}, {n} frames\n\n{table}\n"
    )
    click.echo(table)
