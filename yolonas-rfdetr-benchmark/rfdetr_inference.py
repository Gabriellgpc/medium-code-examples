"""RF-DETR inference demo — runs Nano and Small (384x384) on a single image."""

from pathlib import Path

import click
import cv2
import supervision as sv
from loguru import logger
from rfdetr import RFDETRNano, RFDETRSmall


MODELS = {
    "rfdetr_nano_384": lambda device: RFDETRNano(resolution=384, device=device),
    "rfdetr_small_384": lambda device: RFDETRSmall(resolution=384, device=device),
}


@click.command()
@click.option("--image-path", "-i", required=True, type=click.Path(exists=True), help="Path to input image")
@click.option("--device", "-d", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device to run on")
@click.option("--threshold", "-t", default=0.5, type=float, help="Confidence threshold")
def main(image_path: str, device: str, threshold: float) -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise click.BadParameter(f"Could not read image: {image_path}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for model_key, factory in MODELS.items():
        logger.info(f"Running {model_key} on {device} (threshold={threshold})")
        model = factory(device)
        detections = model.predict(image_path, threshold=threshold)

        class_names = model.class_names
        labels = [
            f"{class_names[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = image.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        stem = Path(image_path).stem
        out_path = output_dir / f"{stem}_{model_key}_{device}.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"  {len(detections)} detections → {out_path}")

        del model


if __name__ == "__main__":
    main()
