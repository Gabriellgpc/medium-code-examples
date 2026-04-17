"""YOLO-NAS inference demo — runs yolo_nas_s on a single image."""

from pathlib import Path

import click
import cv2
import supervision as sv
from loguru import logger
from modern_yolonas import Detector

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


@click.command()
@click.option("--image-path", "-i", required=True, type=click.Path(exists=True), help="Path to input image")
@click.option("--device", "-d", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device to run on")
@click.option("--threshold", "-t", default=0.25, type=float, help="Confidence threshold")
def main(image_path: str, device: str, threshold: float) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise click.BadParameter(f"Could not read image: {image_path}")

    logger.info(f"Running yolo_nas_s on {device} (threshold={threshold})")
    detector = Detector("yolo_nas_s", device=device, conf_threshold=threshold, input_size=384)
    result = detector(image_path)

    detections = sv.Detections(
        xyxy=result.boxes,
        confidence=result.scores,
        class_id=result.class_ids,
    )

    labels = [
        f"{COCO_NAMES.get(cid, str(cid))} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    stem = Path(image_path).stem
    out_path = output_dir / f"{stem}_yolonas_s_{device}.jpg"
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"  {len(detections)} detections → {out_path}")


if __name__ == "__main__":
    main()
