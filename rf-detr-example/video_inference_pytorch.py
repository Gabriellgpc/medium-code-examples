"""Base class for object detection models."""

from pathlib import Path

import click
import cv2
import supervision as sv
from rfdetr import RFDETRBase

from model import load_class_names


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
    "--device",
    "-d",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    help="Device to use for inference",
)
@click.option(
    "--labelmap-filepath",
    "--labelmap",
    "-l",
    type=click.Path(exists=True),
    default=Path(__file__).parent / "coco_labelmap.txt",
    help="Path to the label map file",
)
def main(
    video_source: str, device: str, labelmap_filepath: str
) -> None:
    """Main function to run the object detection model."""

    # Load the model
    class_names = load_class_names(labelmap_filepath)
    detector = RFDETRBase(resolution=280, device=device)

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

        detections = detector.predict(frame, threshold=0.4)
        fps_ticker.tick()

        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(
                detections.class_id, detections.confidence, strict=False
            )
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(
            annotated_frame, detections, labels
        )

        # Draw FPS on the frame
        annotated_frame = draw_fps(annotated_frame, fps_ticker.fps)

        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
