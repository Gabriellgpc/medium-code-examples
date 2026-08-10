"""Run detect → track over a sequence (or video) → annotated output + MOT preds.

Input is either a SoccerNet sequence dir (``img1/`` frames + ``seqinfo.ini``) or a
plain video file. Output is an annotated ``.mp4`` and a MOTChallenge prediction
file (``<tracker>.txt``) ready for ``snt-evaluate``.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import cv2
import supervision as sv
from loguru import logger

from soccernet_tracking_edge.config import OUTPUT_DIR, ir_path
from soccernet_tracking_edge.core import mot, viz
from soccernet_tracking_edge.core.rfdetr import RFDETRDetector
from soccernet_tracking_edge.core.tracking import make_tracker


def _frame_source(source: Path):
    """Yield (frame_index_1based, bgr_frame) from a sequence dir or a video."""
    if source.is_dir():
        frames = mot.list_frames(source)
        for i, fp in enumerate(frames, start=1):
            img = cv2.imread(str(fp))
            if img is not None:
                yield i, img
    else:
        cap = cv2.VideoCapture(str(source))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            yield i, frame
        cap.release()


@click.command()
@click.option("--source", type=click.Path(exists=True, path_type=Path), required=True,
              help="SoccerNet sequence dir (img1/) or a video file.")
@click.option("--tracker", type=click.Choice(["bytetrack", "ocsort"]), default="bytetrack",
              show_default=True)
@click.option("--precision", default="int8_full", show_default=True,
              help="Which detector IR to run (fp32/fp16/int8_woq/int8_full).")
@click.option("--tag", default="det", show_default=True, help="det (COCO) or soccer (fine-tuned).")
@click.option("--device", default="GPU", show_default=True, help="OpenVINO device (GPU=iGPU).")
@click.option("--accurate/--fast", default=True, show_default=True,
              help="ACCURACY vs PERFORMANCE exec mode. INT8-Large on the iGPU needs --accurate.")
@click.option("--threshold", default=0.4, show_default=True, help="Detection confidence gate.")
@click.option("--fps", default=25.0, show_default=True, help="Sequence frame rate.")
@click.option("--max-frames", default=0, show_default=True, help="0 = all frames.")
@click.option("--no-video", is_flag=True, help="Skip writing the annotated video.")
def track(source, tracker, precision, tag, device, accurate, threshold, fps,
          max_frames, no_video) -> None:
    """Detect with RF-DETR + associate with a tracker over the source."""
    from soccernet_tracking_edge.config import SOCCER_TO_LOCAL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = SOCCER_TO_LOCAL if tag == "soccer" else None
    detector = RFDETRDetector(
        ir_path(precision, tag), device=device, threshold=threshold,
        accurate=accurate, mapping=mapping,
    )
    trk = make_tracker(tracker, frame_rate=fps)

    writer = None
    rows: list[tuple[int, int, float, float, float, float]] = []
    fps_mon = sv.FPSMonitor(sample_size=int(fps))
    t0 = time.perf_counter()
    n = 0
    for frame_idx, frame in _frame_source(Path(source)):
        det = detector.detect(frame)
        det = trk.update(det)  # fresh copy with tracker_id — use the return value
        fps_mon.tick()
        n += 1

        if det.tracker_id is not None:
            for box, tid in zip(det.xyxy, det.tracker_id, strict=False):
                x1, y1, x2, y2 = box
                rows.append((frame_idx, int(tid), float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)))

        if not no_video:
            annotated = viz.annotate(frame, det, fps=fps_mon.fps)
            if writer is None:
                h, w = annotated.shape[:2]
                out_mp4 = OUTPUT_DIR / f"track_{tag}_{tracker}_{precision}.mp4"
                writer = cv2.VideoWriter(
                    str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(annotated)

        if max_frames and n >= max_frames:
            break

    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - t0

    pred_path = OUTPUT_DIR / f"pred_{tag}_{tracker}_{precision}.txt"
    mot.write_mot(pred_path, rows)
    logger.info(
        f"{n} frames, {n / elapsed:.1f} FPS end-to-end · {len(rows)} track rows → {pred_path.name}"
    )
