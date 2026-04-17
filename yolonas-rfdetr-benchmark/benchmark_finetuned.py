"""Benchmark fine-tuned RF-DETR Nano and YOLO-NAS-S on VOC-2007 val set."""

import time
from pathlib import Path

import click
import cv2
import fiftyone as fo
import numpy as np
import torch
from loguru import logger
from tabulate import tabulate

RESOLUTION = 256
DATASET_DIR = Path("datasets/coco_animals")
RFDETR_CHECKPOINT = Path("runs/rfdetr_nano_voc")
YOLONAS_CHECKPOINT = Path("runs/yolonas_s_voc")

VOC_CLASSES = ["bird", "cat", "dog", "horse", "sheep"]


def load_val_dataset() -> fo.Dataset:
    """Load VOC val set from YOLO export into FiftyOne."""
    name = "voc2007-finetuned-benchmark"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    return fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        name=name,
    )


def xyxy_to_fo(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    img_w: int,
    img_h: int,
) -> fo.Detections:
    """Convert xyxy boxes to FiftyOne Detections with VOC class names."""
    fo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(class_ids[i])
        fo_dets.append(
            fo.Detection(
                label=VOC_CLASSES[cid] if cid < len(VOC_CLASSES) else str(cid),
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


# --- RF-DETR fine-tuned inference ---


def load_rfdetr_finetuned(device: str):
    """Load fine-tuned RF-DETR Nano."""
    from rfdetr import RFDETRNano

    model = RFDETRNano(num_classes=len(VOC_CLASSES), resolution=RESOLUTION, device=device)

    # Load best checkpoint
    ckpt_path = RFDETR_CHECKPOINT / "checkpoint_best_regular.pth"
    if not ckpt_path.exists():
        # Try other checkpoint names
        for name in ["checkpoint.pth", "checkpoint_best_ema.pth"]:
            alt = RFDETR_CHECKPOINT / name
            if alt.exists():
                ckpt_path = alt
                break

    logger.info(f"  Loading RF-DETR checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if "ema_model" in checkpoint and checkpoint["ema_model"] is not None:
        model.model.model.load_state_dict(checkpoint["ema_model"], strict=False)
        logger.info("  Using EMA weights")
    elif "model" in checkpoint:
        model.model.model.load_state_dict(checkpoint["model"], strict=False)

    return model


def run_rfdetr(model, image_path: str, threshold: float):
    """Run fine-tuned RF-DETR inference."""
    detections = model.predict(image_path, threshold=threshold)
    return detections


# --- YOLO-NAS fine-tuned inference ---


def load_yolonas_finetuned(device: str):
    """Load fine-tuned YOLO-NAS-S."""
    from modern_yolonas import yolo_nas_s

    model = yolo_nas_s(pretrained=False, num_classes=len(VOC_CLASSES))

    ckpt_path = YOLONAS_CHECKPOINT / "last.pt"
    logger.info(f"  Loading YOLO-NAS checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Prefer EMA weights
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        model.load_state_dict(checkpoint["ema"]["ema_state_dict"], strict=False)
        logger.info("  Using EMA weights")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    model.to(device)
    return model


def yolonas_preprocess(image_path: str) -> tuple[torch.Tensor, float, tuple[int, int]]:
    """Preprocess for YOLO-NAS: letterbox + normalize."""
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
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor, scale, (pad_left, pad_top)


def run_yolonas(model, image_path: str, device: str, threshold: float, iou_threshold: float = 0.45):
    """Run fine-tuned YOLO-NAS inference, return (boxes_xyxy, scores, class_ids)."""
    tensor, scale, pad = yolonas_preprocess(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        pred_bboxes, pred_scores = model(tensor)

    boxes = pred_bboxes[0].cpu().numpy()
    scores = pred_scores[0].cpu().numpy()

    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > threshold
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    boxes = boxes[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), scale, pad

    # NMS
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cv2_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(cv2_boxes, max_scores.tolist(), threshold, iou_threshold)
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), scale, pad

    indices = indices.flatten()
    boxes = boxes[indices]
    max_scores = max_scores[indices]
    class_ids = class_ids[indices]

    # Undo letterbox
    pad_left, pad_top = pad
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

    return boxes, max_scores, class_ids, scale, pad


MODEL_CONFIGS = {
    "rfdetr_nano_ft": {
        "display": "RFDETRNano FT 256",
        "type": "rfdetr",
    },
    "yolonas_s_ft": {
        "display": "YOLO-NAS-S FT 256",
        "type": "yolonas",
    },
}


@click.command()
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences")
@click.option("--devices", default="cpu,cuda", help="Comma-separated devices")
def main(threshold: float, warmup: int, devices: str) -> None:
    device_list = [d.strip() for d in devices.split(",")]
    if "cuda" in device_list and not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping")
        device_list = [d for d in device_list if d != "cuda"]

    # Load val dataset
    logger.info("Loading VOC-2007 val set")
    dataset = load_val_dataset()
    logger.info(f"  {len(dataset)} samples loaded")

    warmup_path = dataset.first().filepath
    results_table = []

    for device in device_list:
        for model_key, config in MODEL_CONFIGS.items():
            field_name = f"{model_key}_{device}"
            logger.info(f"Benchmarking {config['display']} on {device}")

            # Load model
            if config["type"] == "rfdetr":
                model = load_rfdetr_finetuned(device)
            else:
                model = load_yolonas_finetuned(device)

            # Warmup
            for _ in range(warmup):
                if config["type"] == "rfdetr":
                    run_rfdetr(model, warmup_path, threshold)
                else:
                    run_yolonas(model, warmup_path, device, threshold)
            if device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for sample in dataset:
                image_path = sample.filepath
                image = cv2.imread(image_path)
                img_h, img_w = image.shape[:2]

                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                if config["type"] == "rfdetr":
                    detections = run_rfdetr(model, image_path, threshold)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    # Convert sv.Detections to FiftyOne
                    fo_dets = []
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        fo_dets.append(fo.Detection(
                            label=VOC_CLASSES[int(detections.class_id[i])] if int(detections.class_id[i]) < len(VOC_CLASSES) else str(int(detections.class_id[i])),
                            bounding_box=[
                                float(x1 / img_w), float(y1 / img_h),
                                float((x2 - x1) / img_w), float((y2 - y1) / img_h),
                            ],
                            confidence=float(detections.confidence[i]),
                        ))
                    sample[field_name] = fo.Detections(detections=fo_dets)
                else:
                    boxes, scores, class_ids, _, _ = run_yolonas(model, image_path, device, threshold)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    sample[field_name] = xyxy_to_fo(boxes, scores, class_ids, img_w, img_h)

                times.append(t1 - t0)
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
                "Device": device,
                "mAP": f"{mAP:.4f}" if mAP is not None else "N/A",
                "Avg (ms)": f"{avg_ms:.1f}",
                "FPS": f"{fps:.1f}",
            })

            logger.info(f"  {config['display']} | {device} | mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("FINE-TUNED BENCHMARK RESULTS (VOC-2007 val)")
    print("=" * 70)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
