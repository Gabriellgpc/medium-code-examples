"""Base class for object detection models."""

import cv2
import loguru
import numpy as np
import openvino as ov
import supervision as sv


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
    """
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    x_min = x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None)
    y_min = y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None)
    x_max = x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None)
    y_max = y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    return (x_min, y_min, x_max, y_max)

class RFDETRDetector:  # noqa: D101

    def __init__(
        self,  # noqa: D107, PLR0913
        model_path: str,
        class_names: list[str],
        input_height: int = 560,
        input_width: int = 560,
        max_detections: int = 300,
        min_confidence: float = 0.5,
        nms_threshold: float = 0.7,
        device="AUTO",
        logger=None,
    ) -> None:
        self.model_path = model_path

        self.logger = logger or loguru.logger

        self.class_names = class_names
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = len(class_names)
        self.max_detections = max_detections
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold

        # random color pallet per class
        # seed for reproducibility
        rng = np.random.default_rng(42)
        self.color_pallet = rng.integers(
            0,
            255,
            size=(self.num_classes, 3),
            dtype=np.uint8,
        ).tolist()

        self.dot_annotator = sv.DotAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.box_corner_annotator = sv.BoxCornerAnnotator(thickness=12)
        self.label_annotator = sv.LabelAnnotator()

        self._load_model(device)

    def _load_model(self, device) -> None:
        """Load the model using onnxruntime."""
        self.logger.info(f"Loading model from {self.model_path}")

        self.core = ov.Core()
        self.logger.info(f"Avaible Devices detected: {self.core.get_available_devices()}")

        # check if device is available, show a warning message and use CPU
        if device not in self.core.get_available_devices():
            self.logger.warning(
                f"Device {device} not available, using CPU instead. "
                "Please check the available devices.",
            )
            device = "CPU"

        self.compiled_model = self.core.compile_model(self.model_path, device)
        self.logger.info(f"Model {self.compiled_model}")
        self.output_layer_dets = self.compiled_model.output(0)
        self.output_layer_labels = self.compiled_model.output(1)

        self.logger.debug(f"Output layer: {self.output_layer_dets}")
        self.logger.debug(f"Output layer: {self.output_layer_labels}")


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
            if i == self.max_detections:
                break

            # Convert cxcywh to xyxy
            x_min, y_min, x_max, y_max = convert_cxcywh_to_xyxy(det)

            logits = raw_labels[i]
            class_id = np.argmax(logits)
            confidence = softmax(logits)[class_id]

            # When running in GPU mode,
            # empty predictions in the output have class_id of -1
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

    def _apply_nms(self, detections: sv.Detections) -> sv.Detections:
        """Apply non-maximum suppression to the detections."""
        return (
            detections.with_nms(threshold=self.nms_threshold)
            if detections is not None
            else None
        )

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
        # Apply NMS
        detections = self._apply_nms(detections)
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

        """
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


def main() -> None:
    """Main function to run the object detection model."""
    # Load the model
    model_path = "model/rf_detr.onnx"
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane"]
    detector = RFDETRDetector(
        model_path=model_path,
        class_names=class_names,
        device="AUTO",
    )

    # Load an image
    image = cv2.imread("image.jpg")
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

