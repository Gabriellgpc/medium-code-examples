"""Create side-by-side detection comparison images and MP4 clips."""

from pathlib import Path

import click
import cv2
import numpy as np
import supervision as sv
import time
import torch
from loguru import logger
from modern_yolonas import Detector
from rfdetr import RFDETRNano

RESOLUTION = 256
OUTPUT_DIR = Path("output/comparison")

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

# Colors
RFDETR_COLOR = (46, 139, 87)     # green
YOLONAS_COLOR = (220, 60, 60)    # red
HEADER_BG = (30, 30, 30)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_header(frame, text, fps, color, width):
    """Draw a model header bar on top of a frame."""
    bar_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), HEADER_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, text, (10, 28), FONT, 0.7, color, 2, cv2.LINE_AA)
    fps_text = f"{fps:.1f} FPS"
    tw = cv2.getTextSize(fps_text, FONT, 0.6, 2)[0][0]
    cv2.putText(frame, fps_text, (width - tw - 10, 28), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    return frame


def run_rfdetr(model, image_path, threshold):
    """Run RF-DETR and return (annotated_frame, fps, num_dets)."""
    t0 = time.perf_counter()
    detections = model.predict(image_path, threshold=threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0

    image = cv2.imread(image_path)
    class_names = model.class_names
    labels = [
        f"{class_names[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    box_ann = sv.BoxAnnotator(color=sv.Color.from_hex("#2E8B57"), thickness=2)
    label_ann = sv.LabelAnnotator(color=sv.Color.from_hex("#2E8B57"), text_scale=0.4, text_padding=3)
    annotated = box_ann.annotate(scene=image.copy(), detections=detections)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated, fps, len(detections)


def run_yolonas(detector, image_path, threshold):
    """Run YOLO-NAS and return (annotated_frame, fps, num_dets)."""
    t0 = time.perf_counter()
    result = detector(image_path)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0

    image = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=result.boxes,
        confidence=result.scores,
        class_id=result.class_ids,
    )

    # Filter by threshold
    mask = detections.confidence >= threshold
    detections = detections[mask]

    labels = [
        f"{COCO_NAMES.get(cid, str(cid))} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    box_ann = sv.BoxAnnotator(color=sv.Color.from_hex("#DC3C3C"), thickness=2)
    label_ann = sv.LabelAnnotator(color=sv.Color.from_hex("#DC3C3C"), text_scale=0.4, text_padding=3)
    annotated = box_ann.annotate(scene=image.copy(), detections=detections)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated, fps, len(detections)


def create_comparison(rfdetr_frame, yolonas_frame, rfdetr_fps, yolonas_fps,
                      rfdetr_dets, yolonas_dets, target_h=480):
    """Create side-by-side comparison image."""
    h1, w1 = rfdetr_frame.shape[:2]
    h2, w2 = yolonas_frame.shape[:2]

    # Resize both to same height
    scale1 = target_h / h1
    scale2 = target_h / h2
    r_frame = cv2.resize(rfdetr_frame, (int(w1 * scale1), target_h))
    y_frame = cv2.resize(yolonas_frame, (int(w2 * scale2), target_h))

    rw = r_frame.shape[1]
    yw = y_frame.shape[1]

    # Draw headers
    r_frame = draw_header(r_frame, f"RF-DETR Nano ({rfdetr_dets} dets)", rfdetr_fps, RFDETR_COLOR, rw)
    y_frame = draw_header(y_frame, f"YOLO-NAS-S ({yolonas_dets} dets)", yolonas_fps, YOLONAS_COLOR, yw)

    # Separator
    sep = np.full((target_h, 3, 3), 200, dtype=np.uint8)

    # Concatenate
    combined = np.hstack([r_frame, sep, y_frame])
    return combined


@click.command()
@click.option("--image-dir", "-i", default=None, help="Directory with images (default: COCO val)")
@click.option("--device", "-d", default="cuda", type=click.Choice(["cpu", "cuda"]))
@click.option("--threshold", "-t", default=0.35, type=float)
@click.option("--max-images", "-n", default=20, type=int, help="Max images to process")
@click.option("--video-fps", default=2, type=int, help="FPS for output MP4 (slideshow speed)")
def main(image_dir, device, threshold, max_images, video_fps):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find images
    if image_dir:
        image_paths = sorted(Path(image_dir).glob("*.*"))
    else:
        # Use COCO val images
        coco_dir = Path.home() / "fiftyone" / "coco-2017" / "validation" / "data"
        if not coco_dir.exists():
            logger.error(f"COCO images not found at {coco_dir}. Pass --image-dir.")
            return
        image_paths = sorted(coco_dir.glob("*.jpg"))

    image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_paths = image_paths[:max_images]
    logger.info(f"Processing {len(image_paths)} images on {device}")

    # Load models
    logger.info("Loading RF-DETR Nano...")
    rfdetr = RFDETRNano(resolution=RESOLUTION, device=device)

    logger.info("Loading YOLO-NAS-S...")
    yolonas = Detector("yolo_nas_s", device=device, input_size=RESOLUTION)

    # Warmup
    warmup_path = str(image_paths[0])
    for _ in range(3):
        rfdetr.predict(warmup_path, threshold=threshold)
        yolonas(warmup_path)

    # Process images
    comparison_frames = []
    for i, img_path in enumerate(image_paths):
        img_str = str(img_path)

        r_frame, r_fps, r_dets = run_rfdetr(rfdetr, img_str, threshold)
        y_frame, y_fps, y_dets = run_yolonas(yolonas, img_str, threshold)

        combined = create_comparison(r_frame, y_frame, r_fps, y_fps, r_dets, y_dets)
        comparison_frames.append(combined)

        # Save individual image
        out_path = OUTPUT_DIR / f"compare_{i:04d}.jpg"
        cv2.imwrite(str(out_path), combined)

        logger.info(f"  [{i+1}/{len(image_paths)}] RF-DETR: {r_fps:.1f} FPS ({r_dets} dets) | "
                     f"YOLO-NAS: {y_fps:.1f} FPS ({y_dets} dets)")

    # Create MP4 slideshow
    if comparison_frames:
        h, w = comparison_frames[0].shape[:2]
        # Ensure all frames are the same size
        target_w = w
        target_h = h

        video_path = OUTPUT_DIR / f"comparison_{device}_{RESOLUTION}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (target_w, target_h))

        for frame in comparison_frames:
            fh, fw = frame.shape[:2]
            if fw != target_w or fh != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
            # Write each frame multiple times for longer display
            for _ in range(max(1, video_fps)):
                writer.write(frame)

        writer.release()
        logger.info(f"Video saved: {video_path} ({len(comparison_frames)} frames)")

    logger.info(f"Images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
