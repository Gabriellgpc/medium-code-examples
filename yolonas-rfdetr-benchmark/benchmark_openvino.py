"""Benchmark OpenVINO models (RF-DETR Nano, YOLO-NAS-S) on Intel iGPU."""

import time
from pathlib import Path

import click
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import openvino as ov
import torchvision.transforms.functional as F
from loguru import logger
from PIL import Image
from tabulate import tabulate

RESOLUTION = 256
MODELS_DIR = Path("models")

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

# RF-DETR class names (COCO 91-class mapping, loaded once at startup)
_rfdetr_class_names = None


def get_rfdetr_class_names() -> dict:
    """Load RF-DETR class name mapping from the model."""
    global _rfdetr_class_names
    if _rfdetr_class_names is None:
        from rfdetr import RFDETRNano

        model = RFDETRNano(resolution=RESOLUTION, device="cpu")
        _rfdetr_class_names = dict(model.class_names)
        del model
    return _rfdetr_class_names


# --- Preprocessing ---


def rfdetr_preprocess(image_path: str) -> np.ndarray:
    """Preprocess for RF-DETR: resize + ImageNet normalization."""
    img = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = F.resize(tensor, (RESOLUTION, RESOLUTION))
    return tensor.unsqueeze(0).numpy()


def yolonas_preprocess(image_path: str) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess for YOLO-NAS: letterbox + /255. Returns (tensor, scale, (pad_l, pad_t))."""
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


def rfdetr_postprocess(
    outputs: list, img_w: int, img_h: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RF-DETR OpenVINO outputs to (boxes_xyxy, scores, class_ids)."""
    pred_boxes = outputs[0][0]  # (300, 4) cxcywh normalized
    pred_logits = outputs[1][0]  # (300, 91) logits

    # Sigmoid
    scores = 1.0 / (1.0 + np.exp(-pred_logits))

    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > threshold
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = pred_boxes[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # cxcywh -> xyxy, denormalize
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return boxes_xyxy, max_scores, class_ids


def yolonas_postprocess(
    outputs: list,
    img_w: int,
    img_h: int,
    scale: float,
    pad: tuple[int, int],
    threshold: float,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert YOLO-NAS OpenVINO outputs to (boxes_xyxy, scores, class_ids)."""
    pred_bboxes = outputs[0][0]  # (N, 4) xyxy in input resolution
    pred_scores = outputs[1][0]  # (N, 80)

    max_scores = pred_scores.max(axis=1)
    class_ids = pred_scores.argmax(axis=1)

    mask = max_scores > threshold
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = pred_bboxes[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # NMS
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cv2_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(
        cv2_boxes, max_scores.tolist(), threshold, iou_threshold
    )
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    indices = indices.flatten()
    boxes = boxes[indices]
    max_scores = max_scores[indices]
    class_ids = class_ids[indices]

    # Undo letterbox: remove padding offset, then scale to original
    pad_left, pad_top = pad
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

    return boxes, max_scores, class_ids


# --- FiftyOne conversion ---


def xyxy_to_fiftyone(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict,
    img_w: int,
    img_h: int,
) -> fo.Detections:
    """Convert xyxy detections to fo.Detections."""
    fo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        fo_dets.append(
            fo.Detection(
                label=class_names.get(int(class_ids[i]), str(int(class_ids[i]))),
                bounding_box=[
                    float(x1 / img_w),
                    float(y1 / img_h),
                    float((x2 - x1) / img_w),
                    float((y2 - y1) / img_h),
                ],
                confidence=float(scores[i]),
            )
        )
    return fo.Detections(detections=fo_dets)


# --- Benchmark ---

PRECISIONS = ["FP32", "FP16", "INT8"]

MODEL_CONFIGS = {
    "rfdetr_nano_256": {
        "display": "RFDETRNano 256",
        "model_dir": MODELS_DIR / "rfdetr_nano_256",
        "type": "rfdetr",
    },
    "yolonas_s_256": {
        "display": "YOLO-NAS-S 256",
        "model_dir": MODELS_DIR / "yolonas_s_256",
        "type": "yolonas",
    },
}


@click.command()
@click.option("--max-samples", "-n", default=100, type=int, help="Number of COCO val samples")
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences to discard")
@click.option("--device", "-d", default="GPU", help="OpenVINO device (GPU for iGPU, CPU)")
def main(max_samples: int, threshold: float, warmup: int, device: str) -> None:
    core = ov.Core()

    available = core.available_devices
    logger.info(f"Available OpenVINO devices: {available}")
    if device not in available:
        logger.error(f"Device '{device}' not available. Choose from: {available}")
        return

    # Load RF-DETR class names
    rfdetr_names = get_rfdetr_class_names()

    # Load COCO validation
    logger.info(f"Loading COCO-2017 validation ({max_samples} samples)")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=max_samples,
        dataset_name=f"coco-2017-val-ov-benchmark-{max_samples}",
    )

    # Warmup image
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
                logger.error(f"  Failed to compile {config['display']} {precision} on {device}: {e}")
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

                # Preprocess
                if config["type"] == "rfdetr":
                    input_tensor = rfdetr_preprocess(image_path)
                    extra = {}
                else:
                    input_tensor, scale, pad = yolonas_preprocess(image_path)
                    extra = {"scale": scale, "pad": pad}

                # Timed inference
                t0 = time.perf_counter()
                result = compiled(input_tensor)
                t1 = time.perf_counter()
                times.append(t1 - t0)

                # Postprocess
                outputs = [result[compiled.output(i)] for i in range(len(result))]
                if config["type"] == "rfdetr":
                    boxes, scores, class_ids = rfdetr_postprocess(
                        outputs, img_w, img_h, threshold
                    )
                    class_names = rfdetr_names
                else:
                    boxes, scores, class_ids = yolonas_postprocess(
                        outputs,
                        img_w,
                        img_h,
                        extra["scale"],
                        extra["pad"],
                        threshold,
                    )
                    class_names = COCO_NAMES

                fo_dets = xyxy_to_fiftyone(
                    boxes, scores, class_ids, class_names, img_w, img_h
                )
                sample[field_name] = fo_dets
                sample.save()

            avg_ms = (sum(times) / len(times)) * 1000
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            # Evaluate mAP
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

            logger.info(
                f"  {config['display']} {precision} | {device} | "
                f"mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS"
            )

            del compiled

    # Print final table
    print("\n" + "=" * 75)
    print(f"OPENVINO BENCHMARK RESULTS (device={device})")
    print("=" * 75)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
