"""Export RF-DETR to ONNX, then convert to OpenVINO IR (FP32 + FP16).

RF-DETR owns its ONNX export (``model.export()`` writes ``inference_model.onnx``
with the square shape baked in by ``resolution=``); we then ``ov.convert_model``
that ONNX. The IR is already static, which is what the iGPU wants — do not
re-open it as dynamic. INT8 is a separate step (``snt-quantize``).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import openvino as ov
from loguru import logger

from soccernet_tracking_edge.config import MODELS_DIR, RFDETR_RESOLUTION, ir_path

# RF-DETR variant classes keyed by name (all auto-download COCO weights unless a
# fine-tuned checkpoint is passed via --weights).
_VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase",
             "medium": "RFDETRMedium", "large": "RFDETRLarge",
             # patch_size=14 original Large — matches the fine-tuned SoccerNet ckpt.
             "large-deprecated": "RFDETRLargeDeprecated"}


@click.command()
@click.option("--variant", type=click.Choice(list(_VARIANTS)), default="nano", show_default=True)
@click.option("--weights", type=click.Path(exists=True, path_type=Path), default=None,
              help="Fine-tuned checkpoint (.pth). Omit for pretrained COCO weights.")
@click.option("--tag", default="det", show_default=True,
              help="IR name tag: 'det' (COCO) or 'soccer' (fine-tuned).")
@click.option("--resolution", default=RFDETR_RESOLUTION, show_default=True, help="Square input.")
@click.option("--opset", default=17, show_default=True, help="ONNX opset for RF-DETR export.")
def export(variant: str, weights: Path | None, tag: str, resolution: int, opset: int) -> None:
    """Export an RF-DETR variant → ONNX → OpenVINO IR (fp32, fp16)."""
    import rfdetr as _rfdetr

    cls = getattr(_rfdetr, _VARIANTS[variant])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / f"rfdetr_{tag}.onnx"

    kw: dict = {"resolution": resolution, "device": "cpu"}
    if weights is not None:
        kw["pretrain_weights"] = str(weights)  # loads fine-tuned head (num_classes auto)
    logger.info(
        f"Loading RF-DETR-{variant} (resolution={resolution}"
        f"{', fine-tuned' if weights else ', COCO'}) and exporting ONNX…"
    )
    model = cls(**kw)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp, opset_version=opset, verbose=False)
        # rfdetr names the file by variant (e.g. "rfdetr-nano.onnx"); older
        # versions used "inference_model.onnx". Just take whatever .onnx it wrote.
        produced = sorted(Path(tmp).glob("*.onnx"))
        if not produced:
            raise click.ClickException(f"RF-DETR wrote no .onnx into {tmp}")
        shutil.move(str(produced[0]), str(onnx_path))
    logger.info(f"ONNX → {onnx_path}")

    logger.info("Converting ONNX → OpenVINO IR…")
    ov_model = ov.convert_model(str(onnx_path))

    fp32 = ir_path("fp32", tag)
    fp16 = ir_path("fp16", tag)
    ov.save_model(ov_model, str(fp32))
    ov.save_model(ov_model, str(fp16), compress_to_fp16=True)
    logger.info(f"Saved IR: {fp32.name}, {fp16.name}")

    # Sanity: compile on CPU and run one zero forward; RF-DETR must emit ≥2
    # outputs (boxes + logits). Cheap guard against a broken graph capture.
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(fp32)), "CPU")
    dummy = np.zeros((1, 3, resolution, resolution), dtype=np.float32)
    outs = compiled(dummy)
    n_out = len(outs)
    if n_out < 2:
        raise click.ClickException(f"Expected ≥2 outputs (boxes, logits), got {n_out}")
    logger.info(f"Sanity OK: {n_out} outputs, shapes {[tuple(o.shape) for o in outs.values()]}")
