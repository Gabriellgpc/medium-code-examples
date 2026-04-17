"""Evaluate all fine-tuned models on VOC val set using FiftyOne COCO eval.

Standardized evaluation: same dataset, same metric, same threshold for all models.
Models: RF-DETR Nano FT, modern-yolonas FT, Deci SG FT
"""

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
CLASSES = ["bird", "cat", "dog", "horse", "sheep"]


def load_val_dataset() -> fo.Dataset:
    """Load VOC val set from YOLO export into FiftyOne."""
    name = "coco-animals-eval-all"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    return fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        name=name,
    )


def xyxy_to_fo(boxes, scores, class_ids, img_w, img_h) -> fo.Detections:
    fo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(class_ids[i])
        fo_dets.append(fo.Detection(
            label=CLASSES[cid] if cid < len(CLASSES) else str(cid),
            bounding_box=[
                float(x1 / img_w), float(y1 / img_h),
                float((x2 - x1) / img_w), float((y2 - y1) / img_h),
            ],
            confidence=float(scores[i]),
        ))
    return fo.Detections(detections=fo_dets)


# --- RF-DETR fine-tuned ---

def load_rfdetr_ft(device):
    from rfdetr import RFDETRNano
    model = RFDETRNano(num_classes=len(CLASSES), resolution=RESOLUTION, device=device)
    ckpt_path = Path("runs/rfdetr_nano_voc/checkpoint_best_regular.pth")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "ema_model" in checkpoint and checkpoint["ema_model"] is not None:
        model.model.model.load_state_dict(checkpoint["ema_model"], strict=False)
    elif "model" in checkpoint:
        model.model.model.load_state_dict(checkpoint["model"], strict=False)
    return model


def predict_rfdetr(model, image_path, threshold):
    detections = model.predict(image_path, threshold=threshold)
    return detections.xyxy, detections.confidence, detections.class_id


# --- modern-yolonas fine-tuned ---

def load_modern_yolonas_ft(device):
    from modern_yolonas import yolo_nas_s
    model = yolo_nas_s(pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load("runs/yolonas_s_voc/last.pt", map_location=device, weights_only=False)
    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"]["ema_state_dict"], strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model


def _letterbox(image, target_size, center=True, normalize=True):
    """Letterbox resize. Returns (tensor, scale, pad_left, pad_top)."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    if center:
        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
    else:
        # SG uses top-left (corner) padding
        pad_top = 0
        pad_left = 0
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    rgb = canvas[:, :, ::-1].copy()
    chw = rgb.transpose(2, 0, 1).astype(np.float32)
    if normalize:
        chw /= 255.0
    return chw, scale, pad_left, pad_top


def predict_yolonas(model, image_path, device, threshold):
    """Inference for modern-yolonas (center pad, /255)."""
    image = cv2.imread(image_path)
    chw, scale, pad_left, pad_top = _letterbox(image, RESOLUTION, center=True, normalize=True)
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_bboxes, pred_scores = model(tensor)

    return _postprocess(pred_bboxes, pred_scores, scale, pad_left, pad_top, threshold)



def _postprocess(pred_bboxes, pred_scores, scale, pad_left, pad_top, threshold):
    boxes = pred_bboxes[0].cpu().numpy()
    scores = pred_scores[0].cpu().numpy()
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    mask = max_scores > threshold
    boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # NMS
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cv2_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(cv2_boxes, max_scores.tolist(), threshold, 0.5)
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    indices = indices.flatten()
    boxes, max_scores, class_ids = boxes[indices], max_scores[indices], class_ids[indices]

    # Undo letterbox
    boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

    return boxes, max_scores, class_ids


@click.command()
@click.option("--threshold", "-t", default=0.25, type=float, help="Confidence threshold")
@click.option("--device", "-d", default="cuda", help="Device")
def main(threshold: float, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")

    logger.info(f"Evaluating all fine-tuned models (threshold={threshold}, device={device})")
    dataset = load_val_dataset()
    logger.info(f"  {len(dataset)} val samples loaded")

    models_to_eval = {
        "rfdetr_nano_ft": {
            "display": "RFDETRNano FT",
            "load": lambda: load_rfdetr_ft(device),
            "predict": lambda m, p: predict_rfdetr(m, p, threshold),
        },
        "modern_yolonas_ft": {
            "display": "modern-yolonas FT",
            "load": lambda: load_modern_yolonas_ft(device),
            "predict": lambda m, p: predict_yolonas(m, p, device, threshold),
        },
    }

    results_table = []

    for model_key, config in models_to_eval.items():
        field_name = f"{model_key}_{device}"
        logger.info(f"Evaluating {config['display']} on {device}")

        try:
            model = config["load"]()
        except Exception as e:
            logger.error(f"  Failed to load: {e}")
            continue

        # Warmup
        warmup_path = dataset.first().filepath
        for _ in range(3):
            config["predict"](model, warmup_path)
        if device == "cuda":
            torch.cuda.synchronize()

        # Run inference on all val samples
        times = []
        for sample in dataset:
            image_path = sample.filepath
            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            boxes, scores, class_ids = config["predict"](model, image_path)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

            sample[field_name] = xyxy_to_fo(boxes, scores, class_ids, img_w, img_h)
            sample.save()

        avg_ms = (sum(times) / len(times)) * 1000
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        # FiftyOne COCO eval
        eval_results = dataset.evaluate_detections(
            field_name,
            gt_field="ground_truth",
            method="coco",
            eval_key=field_name,
            compute_mAP=True,
        )
        mAP = eval_results.mAP()

        # Extract mAP@0.50 specifically
        iou_50_idx = np.where(np.isclose(eval_results.iou_threshs, 0.5))[0]
        if len(iou_50_idx) > 0:
            ap_at_50 = eval_results.precision[iou_50_idx[0]]
            ap_at_50 = ap_at_50[ap_at_50 > -1]
            mAP_50 = float(np.mean(ap_at_50)) if ap_at_50.size > 0 else -1
        else:
            mAP_50 = -1

        results_table.append({
            "Model": config["display"],
            "Device": device,
            "mAP@0.50": f"{mAP_50:.4f}" if mAP_50 >= 0 else "N/A",
            "mAP@0.50:0.95": f"{mAP:.4f}" if mAP is not None and mAP >= 0 else "N/A",
            "Avg (ms)": f"{avg_ms:.1f}",
            "FPS": f"{fps:.1f}",
        })

        logger.info(f"  {config['display']} | mAP@50={mAP_50:.4f} | mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print("FINE-TUNED MODELS — STANDARDIZED FIFTYONE COCO EVAL")
    print(f"  Dataset: COCO Animals val ({len(dataset)} samples, {len(CLASSES)} classes)")
    print(f"  Threshold: {threshold}, Device: {device}")
    print("=" * 75)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
