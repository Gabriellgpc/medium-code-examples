"""Export fine-tuned RF-DETR Nano and YOLO-NAS-S to OpenVINO IR (FP32, FP16, INT8)."""

from pathlib import Path

import click
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from loguru import logger
from PIL import Image

RESOLUTION = 256
DATASET_DIR = Path("datasets/coco_animals")
MODELS_DIR = Path("models_finetuned")
NUM_CLASSES = 5


def rfdetr_preprocess(image_path: str) -> np.ndarray:
    """Preprocess for RF-DETR: resize + ImageNet normalization."""
    img = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = F.resize(tensor, (RESOLUTION, RESOLUTION))
    return tensor.unsqueeze(0).numpy()


def yolonas_preprocess(image_path: str) -> np.ndarray:
    """Preprocess for YOLO-NAS: letterbox + /255 normalization."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    scale = min(RESOLUTION / h, RESOLUTION / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((RESOLUTION, RESOLUTION, 3), 114, dtype=np.uint8)
    pad_top = (RESOLUTION - new_h) // 2
    pad_left = (RESOLUTION - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    rgb = canvas[:, :, ::-1].copy()
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, 0)


def export_rfdetr_onnx(output_path: Path) -> None:
    """Export fine-tuned RFDETRNano to ONNX."""
    from rfdetr import RFDETRNano

    logger.info("Exporting fine-tuned RFDETRNano to ONNX...")
    model = RFDETRNano(num_classes=NUM_CLASSES, resolution=RESOLUTION, device="cpu")

    # Load fine-tuned weights
    ckpt_dir = Path("runs/rfdetr_nano_voc")
    for name in ["checkpoint_best_regular.pth", "checkpoint.pth", "checkpoint_best_ema.pth"]:
        ckpt_path = ckpt_dir / name
        if ckpt_path.exists():
            break

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "ema_model" in checkpoint and checkpoint["ema_model"] is not None:
        model.model.model.load_state_dict(checkpoint["ema_model"], strict=False)
    elif "model" in checkpoint:
        model.model.model.load_state_dict(checkpoint["model"], strict=False)

    # Switch to export mode
    torch_model = model.model.model
    torch_model.eval()
    torch_model.export()

    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        torch_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["dets", "labels"],
        opset_version=17,
        dynamo=False,
    )

    del model
    logger.info(f"  Saved to {output_path}")


def export_yolonas_onnx(output_path: Path) -> None:
    """Export fine-tuned YOLO-NAS-S to ONNX."""
    from modern_yolonas import yolo_nas_s

    logger.info("Exporting fine-tuned YOLO-NAS-S to ONNX...")
    model = yolo_nas_s(pretrained=False, num_classes=NUM_CLASSES).eval()

    # Load fine-tuned weights
    ckpt_path = Path("runs/yolonas_s_voc/last.pt")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        model.load_state_dict(checkpoint["ema"]["ema_state_dict"], strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Fuse RepVGG blocks
    for module in model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()

    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["pred_bboxes", "pred_scores"],
        opset_version=17,
        dynamo=False,
    )

    del model
    logger.info(f"  Saved to {output_path}")


def convert_and_quantize(
    onnx_path: Path,
    output_dir: Path,
    preprocess_fn,
    calibration_images: list[str],
    int8_ignored_scope: list[str] | None = None,
) -> None:
    """Convert ONNX to OpenVINO IR in FP32, FP16, and INT8."""
    import nncf
    import openvino as ov

    logger.info(f"Converting {onnx_path.name} to OpenVINO IR...")
    ov_model = ov.convert_model(str(onnx_path))

    # FP32
    fp32_dir = output_dir / "FP32"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp32_dir / "model.xml"))
    logger.info(f"  FP32 -> {fp32_dir}")

    # FP16
    fp16_dir = output_dir / "FP16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp16_dir / "model.xml"), compress_to_fp16=True)
    logger.info(f"  FP16 -> {fp16_dir}")

    # INT8 via NNCF
    logger.info("  Quantizing to INT8...")

    def calibration_data():
        for img_path in calibration_images:
            yield preprocess_fn(img_path)

    quantize_kwargs = {"preset": nncf.QuantizationPreset.PERFORMANCE}
    if int8_ignored_scope:
        quantize_kwargs["ignored_scope"] = nncf.IgnoredScope(patterns=int8_ignored_scope)

    calibration_dataset = nncf.Dataset(calibration_data())
    int8_model = nncf.quantize(ov_model, calibration_dataset, **quantize_kwargs)

    int8_dir = output_dir / "INT8"
    int8_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(int8_model, str(int8_dir / "model.xml"))
    logger.info(f"  INT8 -> {int8_dir}")


def get_calibration_images() -> list[str]:
    """Get VOC val images for INT8 calibration."""
    img_dir = DATASET_DIR / "images" / "val"
    return [str(p) for p in sorted(img_dir.glob("*.*"))]


@click.command()
def main() -> None:
    cal_images = get_calibration_images()
    if not cal_images:
        logger.error("No calibration images found. Run download_dataset.py first.")
        return
    logger.info(f"Using {len(cal_images)} calibration images")

    # RF-DETR fine-tuned
    rfdetr_onnx = MODELS_DIR / "rfdetr_nano_ft_256" / "model.onnx"
    export_rfdetr_onnx(rfdetr_onnx)
    convert_and_quantize(rfdetr_onnx, MODELS_DIR / "rfdetr_nano_ft_256", rfdetr_preprocess, cal_images)

    # YOLO-NAS fine-tuned
    yolonas_onnx = MODELS_DIR / "yolonas_s_ft_256" / "model.onnx"
    export_yolonas_onnx(yolonas_onnx)
    convert_and_quantize(
        yolonas_onnx, MODELS_DIR / "yolonas_s_ft_256", yolonas_preprocess, cal_images,
        int8_ignored_scope=["/heads/.*"],
    )

    logger.info("All fine-tuned model exports complete!")


if __name__ == "__main__":
    main()
