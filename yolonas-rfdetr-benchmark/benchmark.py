"""Benchmark RF-DETR (nano/small) and YOLO-NAS (small) — mAP via FiftyOne + inference speed."""

import time

import click
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
from loguru import logger
from modern_yolonas import Detector
from rfdetr import RFDETRNano, RFDETRSmall
from tabulate import tabulate

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

MODEL_CONFIGS = {
    "rfdetr_nano_384": {
        "display": "RFDETRNano 256",
        "factory": lambda device: RFDETRNano(resolution=256, device=device),
        "type": "rfdetr",
    },
    "yolonas_s": {
        "display": "YOLO-NAS-S 256",
        "factory": lambda device: Detector("yolo_nas_s", device=device, input_size=256),
        "type": "yolonas",
    },
}


def xyxy_to_fiftyone(xyxy: np.ndarray, img_w: int, img_h: int) -> list[list[float]]:
    """Convert xyxy absolute pixels to FiftyOne [x_rel, y_rel, w_rel, h_rel]."""
    bboxes = []
    for x1, y1, x2, y2 in xyxy:
        bboxes.append([
            float(x1 / img_w),
            float(y1 / img_h),
            float((x2 - x1) / img_w),
            float((y2 - y1) / img_h),
        ])
    return bboxes


def run_rfdetr(model, image_path: str, threshold: float):
    """Run RF-DETR and return (detections_sv, class_names_dict)."""
    detections = model.predict(image_path, threshold=threshold)
    return detections, model.class_names


def run_yolonas(model, image_path: str, threshold: float):
    """Run YOLO-NAS and return (boxes, scores, class_ids)."""
    result = model(image_path)
    return result.boxes, result.scores, result.class_ids


def to_fo_detections(
    model_type: str,
    raw_output,
    img_w: int,
    img_h: int,
) -> fo.Detections:
    """Convert model output to fo.Detections."""
    fo_dets = []

    if model_type == "rfdetr":
        detections, class_names = raw_output
        bboxes = xyxy_to_fiftyone(detections.xyxy, img_w, img_h)
        for bbox, cid, conf in zip(bboxes, detections.class_id, detections.confidence):
            fo_dets.append(fo.Detection(
                label=class_names[int(cid)],
                bounding_box=bbox,
                confidence=float(conf),
            ))
    else:  # yolonas
        boxes, scores, class_ids = raw_output
        bboxes = xyxy_to_fiftyone(boxes, img_w, img_h)
        for bbox, cid, conf in zip(bboxes, class_ids, scores):
            fo_dets.append(fo.Detection(
                label=COCO_NAMES.get(int(cid), str(int(cid))),
                bounding_box=bbox,
                confidence=float(conf),
            ))

    return fo.Detections(detections=fo_dets)


@click.command()
@click.option("--max-samples", "-n", default=100, type=int, help="Number of COCO val samples")
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences to discard")
@click.option("--devices", default="cpu,cuda", help="Comma-separated devices to benchmark")
def main(max_samples: int, threshold: float, warmup: int, devices: str) -> None:
    device_list = [d.strip() for d in devices.split(",")]

    # Filter out cuda if not available
    if "cuda" in device_list and not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping cuda device")
        device_list = [d for d in device_list if d != "cuda"]

    if not device_list:
        logger.error("No valid devices available")
        return

    # Load COCO validation dataset
    logger.info(f"Loading COCO-2017 validation ({max_samples} samples)")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=max_samples,
        dataset_name=f"coco-2017-val-benchmark-{max_samples}",
    )

    results_table = []

    for device in device_list:
        for model_key, config in MODEL_CONFIGS.items():
            field_name = f"{model_key}_{device}"
            logger.info(f"Benchmarking {config['display']} on {device}")

            # Instantiate model
            model = config["factory"](device)

            # Warmup
            first_sample = dataset.first()
            warmup_path = first_sample.filepath
            for _ in range(warmup):
                if config["type"] == "rfdetr":
                    run_rfdetr(model, warmup_path, threshold)
                else:
                    run_yolonas(model, warmup_path, threshold)
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
                    raw = run_rfdetr(model, image_path, threshold)
                else:
                    raw = (run_yolonas(model, image_path, threshold))

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                times.append(t1 - t0)

                fo_dets = to_fo_detections(config["type"], raw, img_w, img_h)
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
                "Device": device,
                "mAP": f"{mAP:.4f}" if mAP is not None else "N/A",
                "Avg (ms)": f"{avg_ms:.1f}",
                "FPS": f"{fps:.1f}",
            })

            logger.info(f"  {config['display']} | {device} | mAP={mAP:.4f} | {avg_ms:.1f}ms | {fps:.1f} FPS")

            # Cleanup
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Print final comparison table
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(tabulate(results_table, headers="keys", tablefmt="pipe"))
    print()


if __name__ == "__main__":
    main()
