"""Multi-model benchmark: RF-DETR, YOLO-NAS, YOLOX, YOLOv9, RTMDet on COCO val.

Uses the `detectors` framework for YOLOX/YOLOv9/RTMDet/YOLO-NAS and `rfdetr` for RF-DETR.
Evaluates mAP via FiftyOne COCO eval at 384x384 and 256x256.
"""

import time
from pathlib import Path

import click
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
from loguru import logger
from tabulate import tabulate

RESOLUTIONS = [384, 256]


def xyxy_to_fo(boxes, scores, class_ids, class_names, img_w, img_h):
    """Convert xyxy detections to fo.Detections."""
    fo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(class_ids[i])
        fo_dets.append(fo.Detection(
            label=class_names.get(cid, str(cid)) if isinstance(class_names, dict) else (
                class_names[cid] if cid < len(class_names) else str(cid)
            ),
            bounding_box=[
                float(x1 / img_w), float(y1 / img_h),
                float((x2 - x1) / img_w), float((y2 - y1) / img_h),
            ],
            confidence=float(scores[i]),
        ))
    return fo.Detections(detections=fo_dets)


# --- RF-DETR (uses rfdetr package directly) ---

def make_rfdetr(resolution, device):
    from rfdetr import RFDETRNano
    return RFDETRNano(resolution=resolution, device=device)


def predict_rfdetr(model, image_path, threshold):
    detections = model.predict(image_path, threshold=threshold)
    return detections.xyxy, detections.confidence, detections.class_id, model.class_names


# --- detectors framework models ---

def make_detector(name, resolution, device):
    from detectors import Detector
    return Detector(name, device=device, input_size=resolution, pretrained=True)


def predict_detector(det, image_path, threshold):
    from detectors.inference.visualize import COCO_NAMES
    result = det(image_path, conf_threshold=threshold)
    return result.boxes, result.scores, result.class_ids, COCO_NAMES


# --- Model registry ---

MODELS = {
    "rfdetr_nano": {
        "display": "RFDETRNano",
        "type": "rfdetr",
    },
    "yolo_nas_s": {
        "display": "YOLO-NAS-S",
        "type": "detectors",
        "name": "yolo-nas-s",
    },
    "yolox_s": {
        "display": "YOLOX-S",
        "type": "detectors",
        "name": "yolox-s",
    },
    "yolov9_s": {
        "display": "YOLOv9-S",
        "type": "detectors",
        "name": "yolov9-s",
    },
    "rtmdet_tiny": {
        "display": "RTMDet-Tiny",
        "type": "detectors",
        "name": "rtmdet-tiny",
    },
    "rtmdet_s": {
        "display": "RTMDet-S",
        "type": "detectors",
        "name": "rtmdet-s",
    },
}


@click.command()
@click.option("--max-samples", "-n", default=50, type=int, help="COCO val samples")
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences")
@click.option("--devices", default="cpu,cuda", help="Comma-separated devices")
def main(max_samples: int, threshold: float, warmup: int, devices: str) -> None:
    device_list = [d.strip() for d in devices.split(",")]
    if "cuda" in device_list and not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping")
        device_list = [d for d in device_list if d != "cuda"]

    # Load COCO val
    logger.info(f"Loading COCO-2017 validation ({max_samples} samples)")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=max_samples,
        dataset_name=f"coco-2017-val-multi-{max_samples}",
    )

    warmup_path = dataset.first().filepath
    results_table = []

    for resolution in RESOLUTIONS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Resolution: {resolution}x{resolution}")
        logger.info(f"{'='*60}")

        for device in device_list:
            for model_key, config in MODELS.items():
                field_name = f"{model_key}_{resolution}_{device}"
                display = f"{config['display']} {resolution}"

                logger.info(f"Benchmarking {display} on {device}")

                # Instantiate model
                try:
                    if config["type"] == "rfdetr":
                        model = make_rfdetr(resolution, device)
                    else:
                        model = make_detector(config["name"], resolution, device)
                except Exception as e:
                    logger.error(f"  Failed to load {display}: {e}")
                    continue

                # Warmup
                for _ in range(warmup):
                    if config["type"] == "rfdetr":
                        predict_rfdetr(model, warmup_path, threshold)
                    else:
                        predict_detector(model, warmup_path, threshold)
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
                        boxes, scores, class_ids, class_names = predict_rfdetr(model, image_path, threshold)
                    else:
                        boxes, scores, class_ids, class_names = predict_detector(model, image_path, threshold)

                    if device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                    fo_dets = xyxy_to_fo(boxes, scores, class_ids, class_names, img_w, img_h)
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
                    "Model": display,
                    "Device": device,
                    "mAP": f"{mAP:.4f}" if mAP is not None else "N/A",
                    "Avg (ms)": f"{avg_ms:.1f}",
                    "FPS": f"{fps:.1f}",
                })

                logger.info(f"  {display} | {device} | mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS")

                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    # Print final table
    print("\n" + "=" * 75)
    print("MULTI-MODEL BENCHMARK RESULTS")
    print("=" * 75)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
