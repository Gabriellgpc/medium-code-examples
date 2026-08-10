"""NNCF INT8 quantization of the RF-DETR IR — two modes.

- ``weight-only``: ``compress_weights`` — shrinks the model, needs no calibration
  data, but leaves activations in float → **no iGPU compute speedup** (the honest
  contrast the article draws).
- ``full``: ``nncf.quantize`` with a calibration set → activations quantized too →
  the real iGPU speedup. RF-DETR is a transformer, so we pass
  ``ModelType.TRANSFORMER`` (SmoothQuant; protects attention/LayerNorm).

Calibration reuses the *exact same* numpy preprocess as inference, so the
activation statistics match what the model sees at run time.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import click
import cv2
import nncf
import numpy as np
import openvino as ov
from loguru import logger

from soccernet_tracking_edge.config import RFDETR_RESOLUTION, ir_path
from soccernet_tracking_edge.core.rfdetr import preprocess


def _read_input_size(xml: Path) -> int:
    model = ov.Core().read_model(str(xml))
    shape = model.input(0).partial_shape
    if shape.rank.is_static and shape[2].is_static:
        return int(shape[2].get_length())
    return RFDETR_RESOLUTION


def _calibration_tensors(images: list[Path], size: int, num_samples: int):
    for img_path in islice(images, num_samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        yield preprocess(img, size)


@click.group()
def cli() -> None:
    """INT8 quantization (weight-only and full PTQ)."""


@cli.command("weight-only")
@click.option("--tag", default="det", show_default=True, help="det (COCO) or soccer (fine-tuned).")
def weight_only(tag: str) -> None:
    """Weight-only INT8 (no calibration, no iGPU speedup — the baseline)."""
    src = ir_path("fp32", tag)
    dst = ir_path("int8_woq", tag)
    logger.info(f"Weight-only INT8: {src.name} → {dst.name}")
    compressed = nncf.compress_weights(
        ov.Core().read_model(str(src)), mode=nncf.CompressWeightsMode.INT8_ASYM
    )
    ov.save_model(compressed, str(dst))
    logger.info("Done.")


@cli.command("full")
@click.option("--images-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory of football frames for calibration.")
@click.option("--num-samples", default=128, show_default=True)
@click.option("--tag", default="det", show_default=True, help="det (COCO) or soccer (fine-tuned).")
def full(images_dir: Path, num_samples: int, tag: str) -> None:
    """Full INT8 PTQ (activations quantized → the real iGPU speedup)."""
    src = ir_path("fp32", tag)
    dst = ir_path("int8_full", tag)
    size = _read_input_size(src)
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        raise click.ClickException(f"No images found in {images_dir}")
    logger.info(f"Full INT8 PTQ: {src.name} → {dst.name} ({num_samples} calib frames, size {size})")

    # Materialize to a list: with ModelType.TRANSFORMER, NNCF iterates the
    # calibration set more than once (SmoothQuant pass + main statistics pass).
    # A one-shot generator gets exhausted after the first pass and the second
    # sees an empty dataset ("Calibration dataset must not be empty").
    tensors = list(_calibration_tensors(images, size, num_samples))
    calib = nncf.Dataset(tensors, lambda t: t.astype(np.float32))
    quantized = nncf.quantize(
        ov.Core().read_model(str(src)),
        calib,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    ov.save_model(quantized, str(dst))
    logger.info("Done.")
