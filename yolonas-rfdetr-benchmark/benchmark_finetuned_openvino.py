"""Benchmark fine-tuned OpenVINO models on Intel iGPU (VOC-2007 val set)."""

import time
from pathlib import Path

import click
import cv2
import fiftyone as fo
import numpy as np
import openvino as ov
import torchvision.transforms.functional as F
from loguru import logger
from PIL import Image
from tabulate import tabulate

RESOLUTION = 256
DATASET_DIR = Path("datasets/coco_animals")
MODELS_DIR = Path("models_finetuned")

VOC_CLASSES = ["bird", "cat", "dog", "horse", "sheep"]


# --- Preprocessing ---


def rfdetr_preprocess(image_path: str) -> np.ndarray:
    """Preprocess for RF-DETR: resize + ImageNet normalization."""
    img = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = F.resize(tensor, (RESOLUTION, RESOLUTION))
    return tensor.unsqueeze(0).numpy()


def yolonas_preprocess(image_path: str) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess for YOLO-NAS: letterbox + /255."""
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
    return np.expand_dims(chw, 0), scale, (pad_left, pad_top)


# --- Postprocessing ---


def rfdetr_postprocess(outputs, img_w, img_h, threshold):
    """Convert RF-DETR outputs to (boxes_xyxy, scores, class_ids)."""
    pred_boxes = outputs[0][0]
    pred_logits = outputs[1][0]

    scores = 1.0 / (1.0 + np.exp(-pred_logits))
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > threshold
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = pred_boxes[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return boxes_xyxy, max_scores, class_ids


def yolonas_postprocess(outputs, img_w, img_h, scale, pad, threshold, iou_threshold=0.45):
    """Convert YOLO-NAS outputs to (boxes_xyxy, scores, class_ids)."""
    pred_bboxes = outputs[0][0]
    pred_scores = outputs[1][0]

    max_scores = pred_scores.max(axis=1)
    class_ids = pred_scores.argmax(axis=1)

    mask = max_scores > threshold
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = pred_bboxes[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cv2_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(cv2_boxes, max_scores.tolist(), threshold, iou_threshold)
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    indices = indices.flatten()
    boxes = boxes[indices]
    max_scores = max_scores[indices]
    class_ids = class_ids[indices]

    pad_left, pad_top = pad
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

    return boxes, max_scores, class_ids


def xyxy_to_fo(boxes, scores, class_ids, img_w, img_h) -> fo.Detections:
    """Convert to FiftyOne detections with VOC class names."""
    fo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(class_ids[i])
        fo_dets.append(fo.Detection(
            label=VOC_CLASSES[cid] if cid < len(VOC_CLASSES) else str(cid),
            bounding_box=[
                float(x1 / img_w), float(y1 / img_h),
                float((x2 - x1) / img_w), float((y2 - y1) / img_h),
            ],
            confidence=float(scores[i]),
        ))
    return fo.Detections(detections=fo_dets)


# --- Benchmark ---

PRECISIONS = ["FP32", "FP16", "INT8"]

MODEL_CONFIGS = {
    "rfdetr_nano_ft_256": {
        "display": "RFDETRNano FT 256",
        "model_dir": MODELS_DIR / "rfdetr_nano_ft_256",
        "type": "rfdetr",
    },
    "yolonas_s_ft_256": {
        "display": "YOLO-NAS-S FT 256",
        "model_dir": MODELS_DIR / "yolonas_s_ft_256",
        "type": "yolonas",
    },
}


def load_val_dataset() -> fo.Dataset:
    """Load VOC val set from YOLO export."""
    name = "voc2007-finetuned-ov-benchmark"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    return fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        name=name,
    )


@click.command()
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences")
@click.option("--device", "-d", default="GPU.0", help="OpenVINO device")
def main(threshold: float, warmup: int, device: str) -> None:
    core = ov.Core()

    available = core.available_devices
    logger.info(f"Available OpenVINO devices: {available}")
    if device not in available:
        logger.error(f"Device '{device}' not available. Choose from: {available}")
        return

    logger.info("Loading VOC-2007 val set")
    dataset = load_val_dataset()
    logger.info(f"  {len(dataset)} samples loaded")

    warmup_path = dataset.first().filepath
    results_table = []

    for model_key, config in MODEL_CONFIGS.items():
        for precision in PRECISIONS:
            field_name = f"{model_key}_{precision}_{device}".lower().replace(".", "_")
            model_path = config["model_dir"] / precision / "model.xml"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}, skipping")
                continue

            logger.info(f"Benchmarking {config['display']} {precision} on {device}")

            try:
                compiled = core.compile_model(str(model_path), device)
            except RuntimeError as e:
                logger.error(f"  Failed to compile: {e}")
                continue

            # Warmup
            if config["type"] == "rfdetr":
                warmup_input = rfdetr_preprocess(warmup_path)
            else:
                warmup_input, _, _ = yolonas_preprocess(warmup_path)
            for _ in range(warmup):
                compiled(warmup_input)

            # Benchmark
            times = []
            for sample in dataset:
                image_path = sample.filepath
                image = cv2.imread(image_path)
                img_h, img_w = image.shape[:2]

                if config["type"] == "rfdetr":
                    input_tensor = rfdetr_preprocess(image_path)
                    extra = {}
                else:
                    input_tensor, scale, pad = yolonas_preprocess(image_path)
                    extra = {"scale": scale, "pad": pad}

                t0 = time.perf_counter()
                result = compiled(input_tensor)
                t1 = time.perf_counter()
                times.append(t1 - t0)

                outputs = [result[compiled.output(i)] for i in range(len(result))]
                if config["type"] == "rfdetr":
                    boxes, scores, class_ids = rfdetr_postprocess(outputs, img_w, img_h, threshold)
                else:
                    boxes, scores, class_ids = yolonas_postprocess(
                        outputs, img_w, img_h, extra["scale"], extra["pad"], threshold
                    )

                sample[field_name] = xyxy_to_fo(boxes, scores, class_ids, img_w, img_h)
                sample.save()

            avg_ms = (sum(times) / len(times)) * 1000
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            eval_results = dataset.evaluate_detections(
                field_name,
                gt_field="ground_truth",
                method="coco",
                eval_key=field_name.replace("-", "_"),
                compute_mAP=True,
            )
            mAP = eval_results.mAP()

            results_table.append({
                "Model": config["display"],
                "Precision": precision,
                "Device": device,
                "mAP": f"{mAP:.4f}" if mAP is not None else "N/A",
                "Avg (ms)": f"{avg_ms:.1f}",
                "FPS": f"{fps:.1f}",
            })

            logger.info(f"  {config['display']} {precision} | {device} | mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS")
            del compiled

    print("\n" + "=" * 80)
    print(f"FINE-TUNED OPENVINO BENCHMARK RESULTS (device={device}, VOC-2007 val)")
    print("=" * 80)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
