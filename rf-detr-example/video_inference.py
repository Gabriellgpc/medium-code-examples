"""Base class for object detection models."""

from pathlib import Path

import click
import cv2
import supervision as sv
from loguru import logger

from model import RFDETRDetector, load_class_names


def draw_fps(frame, fps):
    """Draw FPS on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame

@click.command()
@click.option(
    "--video-source",
    "-v",
    default="/dev/video0",
    help="Path to the video file or camera index",
)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True),
    default=Path(__file__).parent / "output/ov_model/inference_model.xml",
    help="Path to the model file",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["AUTO", "CPU", "GPU"]),
    default="GPU",
    help="Device to use for inference",
)
@click.option("--labelmap-filepath", "--labelmap", "-l",
              type=click.Path(exists=True),
              default=Path(__file__).parent / "coco_labelmap.txt",
              help="Path to the label map file")
def main(video_source: str,
         model_path: str,
         device: str,
         labelmap_filepath:str) -> None:
    """Main function to run the object detection model."""

    # Load the model
    logger.info(f"Loading model from {model_path}")
    class_names = load_class_names(labelmap_filepath)
    detector = RFDETRDetector(
        model_path=model_path,
        class_names=class_names,
        device=device,
        min_confidence=0.3,
    )


    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError("Video source not found or unable to open.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_ticker = sv.FPSMonitor(sample_size=int(video_fps))

    tracker = sv.ByteTrack(frame_rate=video_fps)
    smoother = sv.DetectionsSmoother()

    while True:
        success, frame = cap.read()

        if not success:
            break

        detections = detector(frame)
        fps_ticker.tick()

        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        annotated_frame = detector.draw_detections(frame, detections)

        # Draw FPS on the frame
        annotated_frame = draw_fps(annotated_frame, fps_ticker.fps)

        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
