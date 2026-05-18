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
from PIL import Image
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

# RF-DETR 1.6.x predict() returns class_id in the raw COCO-91 space
# (pretrained DETR head has 91 output slots with gaps). We map 91 -> COCO-80 name.
COCO_91_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

def build_model_configs(resolution: int) -> dict:
    return {
        "rfdetr_nano": {
            "display": f"RFDETRNano {resolution}",
            "factory": lambda device: RFDETRNano(resolution=resolution, device=device),
            "type": "rfdetr",
        },
        "yolonas_s": {
            "display": f"YOLO-NAS-S {resolution}",
            "factory": lambda device: Detector("yolo_nas_s", device=device, input_size=resolution),
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
    """Run RF-DETR and return (detections_sv,).

    class_names intentionally omitted — we use COCO_91_TO_NAME in to_fo_detections
    to guarantee identical label space for both RF-DETR and YOLO-NAS mAP eval.
    Loads via PIL to guarantee 3-channel RGB (some COCO val images are grayscale).
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    detections = model.predict(img, threshold=threshold)
    return (detections,)


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
        (detections,) = raw_output
        bboxes = xyxy_to_fiftyone(detections.xyxy, img_w, img_h)
        for bbox, cid, conf in zip(bboxes, detections.class_id, detections.confidence):
            label = COCO_91_TO_NAME.get(int(cid))
            if label is None:
                continue
            fo_dets.append(fo.Detection(
                label=label,
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
@click.option("--max-samples", "-n", default=500, type=int, help="Number of COCO val samples")
@click.option("--threshold", "-t", default=0.05, type=float, help="Confidence threshold (low for proper mAP PR-curve coverage)")
@click.option("--warmup", "-w", default=5, type=int, help="Warmup inferences to discard")
@click.option("--devices", default="cpu,cuda", help="Comma-separated devices to benchmark")
@click.option("--resolution", "-r", default=256, type=int, help="Input resolution (square)")
def main(max_samples: int, threshold: float, warmup: int, devices: str, resolution: int) -> None:
    model_configs = build_model_configs(resolution)
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
        for model_key, config in model_configs.items():
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
