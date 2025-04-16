"""Base class for object detection models."""

from pathlib import Path

import click
import cv2
import supervision as sv
from loguru import logger

from model import RFDETRDetector, load_class_names


@click.command()
@click.option(
    "--video-source",
    "-v",
    type=click.Path(exists=True),
    required=True,
    help="Path to the video file or camera index",
)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model file",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["AUTO", "CPU", "GPU"]),
    default="GPU",
    help="Device to use for inference",
)
def main(video_source: str, model_path: str, device: str) -> None:
    """Main function to run the object detection model."""

    # Load the model
    logger.info(f"Loading model from {model_path}")
    class_names_filepath = Path(__file__).parent / "coco_labelmap.txt"
    class_names = load_class_names(class_names_filepath)
    detector = RFDETRDetector(
        model_path=model_path,
        class_names=class_names,
        device=device
    )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError("Video source not found or unable to open.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_ticker = sv.FPSMonitor(sample_size=int(video_fps))
    while True:
        success, frame = cap.read()
        fps_ticker.tick()
        if not success:
            break

        detections = detector(frame)

        annotated_frame = detector.draw_detections(frame, detections)

        # Draw FPS on the frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps_ticker.fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
