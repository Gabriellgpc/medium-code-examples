"""Base class for object detection models."""

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import supervision as sv
from loguru import logger


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def convert_cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center x, center y, width, height (cxcywh) format
    to x_min, y_min, x_max, y_max (xyxy) format using numpy.

    Args:
        x (np.ndarray): Array of shape (..., 4) where each row is [cx, cy, w, h].

    Returns:
        np.ndarray: Array of shape (..., 4) where each row is [x_min, y_min, x_max, y_max].
    """  # noqa: E501
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    x_min = x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None)
    y_min = y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None)
    x_max = x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None)
    y_max = y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    return (x_min, y_min, x_max, y_max)

def load_class_names(class_names_filepath: Path) -> list[str]:
    with class_names_filepath.open("r") as file:
        class_names = file.read().splitlines()
    return class_names


class RFDETRDetector:  # noqa: D101

    def __init__(
        self,  # noqa: D107, PLR0913
        model_path: str,
        class_names: list[str],
        input_height: int = 280,
        input_width: int = 280,
        max_detections: int = 300,
        min_confidence: float = 0.5,
        nms_threshold: float = 0.7,
        device="AUTO"
    ) -> None:
        self.model_path = model_path

        self.class_names = class_names
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = len(class_names)
        self.max_detections = max_detections
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold

        self.dot_annotator = sv.DotAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.box_corner_annotator = sv.BoxCornerAnnotator(thickness=12)
        self.label_annotator = sv.LabelAnnotator()

        self._load_model(device)

    def _load_model(self, device) -> None:
        """Load the model using onnxruntime."""
        logger.debug(f"Loading model from {self.model_path}")

        self.core = ov.Core()
        logger.debug(f"Avaible Devices detected: {self.core.get_available_devices()}")

        # check if device is available, show a warning message and use CPU
        if device not in self.core.get_available_devices():
            logger.warning(
                f"Device {device} not available, using CPU instead. "
                "Please check the available devices.",
            )
            device = "CPU"
        else:
            logger.debug(
                f"Device {device} is available, using it for inference.",
            )

        # Use cache model
        path_to_cache_dir = Path(self.model_path).parent / "cache"
        self.core.set_property({props.cache_dir: path_to_cache_dir})

        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                  hints.execution_mode: hints.ExecutionMode.PERFORMANCE}
        self.compiled_model = self.core.compile_model(self.model_path,
                                                      device,
                                                      config)
        logger.debug(f"Model {self.compiled_model}")
        self.output_layer_dets = self.compiled_model.output(0)
        self.output_layer_labels = self.compiled_model.output(1)

        logger.debug(f"Output layer: {self.output_layer_dets}")
        logger.debug(f"Output layer: {self.output_layer_labels}")


    def _preprocess(self, image):
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        model_input = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA,
        )
        # Convert to RGB
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        # normalize to [0, 1]
        model_input = model_input.astype(np.float32) / 255.0
        # standardize to imagenet mean and std
        model_input = (model_input - mean) / std
        # add batch dim
        model_input = np.expand_dims(model_input, axis=0)
        # Convert to channels first
        model_input = np.transpose(model_input, (0, 3, 1, 2))
        return np.ascontiguousarray(model_input)

    def _filter_by_confidence(
        self,
        scores: np.ndarray,
        class_ids: np.ndarray,
        xyxy: np.ndarray,
        class_names: np.ndarray,
    ) -> dict:
        """Filter detections by confidence level."""
        # Filter detections by confidence level
        keep_indices = np.nonzero(scores >= self.min_confidence)[0]
        out = {
            "scores": scores[keep_indices],
            "class_ids": class_ids[keep_indices],
            "xyxy": xyxy[keep_indices],
            "class_names": class_names[keep_indices],
        }
        return out

    def _model_inference(self, model_input: np.ndarray) -> np.ndarray:
        """Perform inference on the model."""
        raw_outputs = self.compiled_model([model_input])
        return raw_outputs

    def _postprocess(
        self,
        predictions: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> dict:
        """Postprocess the model output to get the detections."""
        xyxy = []
        scores = []
        class_ids = []
        class_names = []

        raw_dets = predictions[self.output_layer_dets][0]
        raw_labels = predictions[self.output_layer_labels][0]

        for i, det in enumerate(raw_dets):
            if i >= self.max_detections:
                break

            # Convert cxcywh to xyxy
            x_min, y_min, x_max, y_max = convert_cxcywh_to_xyxy(det)

            logits = raw_labels[i]
            class_id = np.argmax(logits)
            confidence = softmax(logits)[class_id]

            if class_id < 0 or confidence < 0.0:
                break

            # ignore classes that are not in the class_names list
            if class_id > self.num_classes - 1:
                continue

            x_min = int(x_min * image_width)
            y_min = int(y_min * image_height)
            x_max = int(x_max * image_width)
            y_max = int(y_max * image_height)

            xyxy.append([x_min, y_min, x_max, y_max])
            scores.append(confidence)
            class_ids.append(class_id)
            class_names.append(self.class_names[int(class_id)])

        # Convert to numpy arrays
        return {
            "xyxy": np.array(xyxy),
            "scores": np.array(scores),
            "class_ids": np.array(class_ids, dtype=np.int32),
            "class_names": np.array(class_names, dtype=np.str_),
        }

    def _convert_to_sv_detections(
        self,
        scores: np.ndarray,
        class_ids: np.ndarray,
        xyxy: np.ndarray,
        class_names: np.ndarray,
    ) -> sv.Detections:
        """Convert model outputs to Supervision detections."""
        if len(xyxy) == 0:
            return None

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=scores,
            class_id=class_ids,
            data={"class_name": class_names},
        )
        return detections

    def __call__(self, image: np.ndarray) -> sv.Detections:
        """Perform inference on the given image and return detections.

        Args:
            image (numpy.ndarray): The input image for inference.

        Returns:
            list: The detected objects in the image.

        """
        model_input = self._preprocess(image)
        model_output = self._model_inference(model_input)
        prep_outputs = self._postprocess(model_output, image.shape[1], image.shape[0])
        # Filter detections by confidence level
        prep_outputs = self._filter_by_confidence(**prep_outputs)
        # convert to sv.Detections
        detections = self._convert_to_sv_detections(**prep_outputs)
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: sv.Detections,
    ) -> np.ndarray:
        """Draws detections on the given image.

        This method annotates the provided image with bounding boxes, labels, and corners
        based on the detection results. If no detections are provided, the original image
        is returned unmodified.

        Args:
            image (np.ndarray): The image on which to draw the detections.
            detections (dict): A dictionary containing detection results with keys:
                - "class_name" (list of str): The class names of the detected objects.
                - "confidence" (list of float): The confidence scores of the detections.

        Returns:
            np.ndarray: The annotated image with detections drawn on it.

        """  # noqa: E501
        if detections is None or len(detections) == 0:
            return image

        annotated_image = image.copy()

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(
                detections["class_name"],
                detections.confidence, strict=False,
            )
        ]
        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=detections,
        )
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels,
        )
        annotated_image = self.box_corner_annotator.annotate(
            scene=annotated_image,
            detections=detections,
        )
        return annotated_image  # noqa: RET504

