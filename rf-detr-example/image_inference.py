"""Base class for object detection models."""


import click
import cv2
from loguru import logger

from model import RFDETRDetector, load_class_names


@click.command()
@click.option("--image-path", "-i",
            type=click.Path(exists=True),
            required=True,
            help="Path to the image file")
@click.option("--model-path", "-m",
            type=click.Path(exists=True),
            required=True,
            help="Path to the model file")
@click.option("--device", "-d",
            type=click.Choice(["AUTO", "CPU", "GPU"]),
            default="AUTO",
            help="Device to use for inference")
def main(image_path: str, model_path: str, device: str) -> None:
    """Main function to run the object detection model."""

    # Load the model
    logger.info(f"Loading model from {model_path}")
    class_names = load_class_names(model_path)
    detector = RFDETRDetector(
        model_path=model_path,
        class_names=class_names,
        device=device,
    )

    # Load an image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Perform inference
    detections = detector(image)

    # Draw detections on the image
    annotated_image = detector.draw_detections(image, detections)

    # Show the annotated image
    cv2.imshow("Detections", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
