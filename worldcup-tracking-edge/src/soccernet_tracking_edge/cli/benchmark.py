"""Benchmark the pipeline: detector speed × device, and tracking quality.

Two tables, both saved to ``output/``:

1. **Speed** — {fp32, int8_woq, int8_full} × {CPU, GPU (iGPU), …}: warmup then a
   timed *forward-only* loop → mean / p50 / p90 ms + FPS. This is the part
   quantization accelerates; Python pre/postprocess is excluded.
2. **Tracking quality** — {bytetrack, ocsort} × {fp32 det, int8_full det} →
   MOTA / IDF1 / ID-switches on the sequence. This is the propagation story:
   how the detector's precision moves the *identity* numbers downstream.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
from loguru import logger  # noqa: E402
from tabulate import tabulate  # noqa: E402

from soccernet_tracking_edge.config import OUTPUT_DIR, ir_path  # noqa: E402
from soccernet_tracking_edge.core import mot  # noqa: E402
from soccernet_tracking_edge.core.common import resolve_device  # noqa: E402
from soccernet_tracking_edge.core.rfdetr import RFDETRDetector  # noqa: E402
from soccernet_tracking_edge.core.tracking import make_tracker  # noqa: E402


def _time_forward(precision, device, iters, warmup, size):
    det = RFDETRDetector(ir_path(precision), device=device)
    tensor = np.zeros((1, 3, size, size), dtype=np.float32)
    for _ in range(warmup):
        det.model.forward(tensor)
    lat = []
    for _ in range(iters):
        s = time.perf_counter()
        det.model.forward(tensor)
        lat.append((time.perf_counter() - s) * 1000.0)
    return np.array(lat)


def _speed_table(precisions, devices, iters, warmup, size):
    core = ov.Core()
    available = core.get_available_devices()
    devices = [d for d in devices if d == "CPU" or resolve_device(d, available) != "CPU"]
    rows = []
    for precision in precisions:
        if not ir_path(precision).exists():
            logger.warning(f"skip {precision}: IR missing")
            continue
        for device in devices:
            try:
                lat = _time_forward(precision, device, iters, warmup, size)
            except Exception as exc:  # noqa: BLE001 - a device may fail to compile (e.g. NVIDIA/OpenVINO)
                logger.warning(f"{precision} on {device} failed: {str(exc)[:80]}")
                continue
            rows.append({
                "precision": precision, "device": device,
                "mean_ms": lat.mean(), "p50_ms": np.percentile(lat, 50),
                "p90_ms": np.percentile(lat, 90), "fps": 1000.0 / lat.mean(),
            })
            logger.info(
                f"{precision:10s} {device:5s}  {lat.mean():6.2f} ms  {1000 / lat.mean():5.1f} FPS"
            )
    return rows


def _tracking_table(sequence, trackers, precisions, device, fps, max_frames):
    gt = mot.read_mot(sequence / "gt" / "gt.txt")
    rows = []
    for precision in precisions:
        if not ir_path(precision).exists():
            continue
        det = RFDETRDetector(ir_path(precision), device=device, accurate=True)
        for tname in trackers:
            trk = make_tracker(tname, frame_rate=fps)
            pred: dict[int, list] = {}
            frames = mot.list_frames(sequence)
            if max_frames:
                frames = frames[:max_frames]
            import cv2
            for idx, fp in enumerate(frames, start=1):
                img = cv2.imread(str(fp))
                d = trk.update(det.detect(img))
                if d.tracker_id is not None:
                    for box, tid in zip(d.xyxy, d.tracker_id, strict=False):
                        x1, y1, x2, y2 = box
                        pred.setdefault(idx, []).append(
                            [int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        )
            pred_arr = {f: np.asarray(v, dtype=np.float32) for f, v in pred.items()}
            m = mot.evaluate(gt, pred_arr)
            rows.append({"tracker": tname, "det_precision": precision, **m})
            logger.info(
                f"{tname:9s} {precision:10s}  "
                f"MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} IDSW={m['IDSW']}"
            )
    return rows


def _plot(speed_rows: list[dict], out_png: Path) -> None:
    if not speed_rows:
        return
    precisions = sorted({r["precision"] for r in speed_rows})
    devices = sorted({r["device"] for r in speed_rows})
    x = np.arange(len(precisions))
    width = 0.8 / max(len(devices), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, dev in enumerate(devices):
        vals = [next((r["fps"] for r in speed_rows
                      if r["precision"] == p and r["device"] == dev), 0) for p in precisions]
        ax.bar(x + i * width, vals, width, label=dev)
    ax.set_xticks(x + width * (len(devices) - 1) / 2)
    ax.set_xticklabels(precisions, rotation=15)
    ax.set_ylabel("FPS (forward-only)")
    ax.set_title("RF-DETR-Nano throughput by precision × device")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    logger.info(f"chart → {out_png}")


@click.command()
@click.option("--sequence", type=click.Path(exists=True, path_type=Path), default=None,
              help="SoccerNet sequence dir for the tracking-quality table (optional).")
@click.option("--precisions", default="fp32,int8_woq,int8_full", show_default=True)
@click.option("--devices", default="CPU,GPU", show_default=True)
@click.option("--iters", default=50, show_default=True)
@click.option("--warmup", default=5, show_default=True)
@click.option("--fps", default=25.0, show_default=True)
@click.option("--max-frames", default=300, show_default=True, help="Cap tracking eval frames.")
def benchmark(sequence, precisions, devices, iters, warmup, fps, max_frames) -> None:
    """Speed (device × precision) + tracking quality (tracker × precision)."""
    from soccernet_tracking_edge.config import RFDETR_RESOLUTION

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prec_list = [p.strip() for p in precisions.split(",")]
    dev_list = [d.strip() for d in devices.split(",")]

    logger.info("=== Speed ===")
    speed = _speed_table(prec_list, dev_list, iters, warmup, RFDETR_RESOLUTION)

    track_rows: list[dict] = []
    if sequence:
        logger.info("=== Tracking quality ===")
        track_rows = _tracking_table(
            Path(sequence), ["bytetrack", "ocsort"],
            [p for p in prec_list if p in ("fp32", "int8_full")],
            dev_list[-1] if dev_list else "CPU", fps, max_frames,
        )

    # --- write artifacts ---
    speed_tbl = tabulate(
        [[r["precision"], r["device"], f"{r['mean_ms']:.2f}", f"{r['p50_ms']:.2f}",
          f"{r['p90_ms']:.2f}", f"{r['fps']:.1f}"] for r in speed],
        headers=["precision", "device", "mean ms", "p50 ms", "p90 ms", "FPS"],
        tablefmt="github",
    )
    click.echo(speed_tbl)
    md = ["# Benchmark\n", "## Detector speed (forward-only)\n", speed_tbl, "\n"]
    if track_rows:
        track_tbl = tabulate(
            [[r["tracker"], r["det_precision"], f"{r['MOTA']:.3f}", f"{r['IDF1']:.3f}",
              r["IDSW"], r["FP"], r["FN"]] for r in track_rows],
            headers=["tracker", "det precision", "MOTA", "IDF1", "IDSW", "FP", "FN"],
            tablefmt="github",
        )
        click.echo(track_tbl)
        md += ["## Tracking quality\n", track_tbl, "\n"]

    (OUTPUT_DIR / "benchmark.md").write_text("\n".join(md))
    with (OUTPUT_DIR / "speed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["precision", "device", "mean_ms", "p50_ms", "p90_ms", "fps"]
        )
        w.writeheader()
        w.writerows(speed)
    _plot(speed, OUTPUT_DIR / "latency.png")
    logger.info("Saved output/benchmark.md, speed.csv, latency.png")
