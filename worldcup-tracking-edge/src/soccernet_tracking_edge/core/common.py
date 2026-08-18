"""Shared OpenVINO runtime plumbing.

``OVDetModel`` loads a detection IR, pins a static shape, compiles for the
requested device with iGPU-friendly perf hints, and exposes a raw forward pass.
The RF-DETR-specific preprocessing and decode live in ``rfdetr.py``.

The device resolver and the two iGPU heisenbug comments (per-device cache dir,
ACCURACY vs PERFORMANCE) are carried over verbatim from the seg project — they
were paid for in debugging time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from loguru import logger


def resolve_device(requested: str, available: list[str]) -> str:
    """Resolve an OpenVINO device request against the available devices.

    OpenVINO exposes device *families* like ``GPU`` that map to enumerated
    instances such as ``GPU.0`` / ``GPU.1``. A plain family name is a valid
    device target even though it isn't returned verbatim by
    ``get_available_devices()``. Accept it when any instance of that family
    exists; otherwise fall back to CPU. A naive ``requested in available``
    check silently runs ``--device GPU`` on the CPU.
    """
    if requested == "AUTO":
        return requested
    if any(d == requested or d.startswith(requested + ".") for d in available):
        return requested
    logger.warning(
        f"Device {requested!r} not available, falling back to CPU. Available: {available}"
    )
    return "CPU"


class OVDetModel:
    """Wrap an OpenVINO object-detection IR (RF-DETR).

    Handles device resolution, the compiled-model cache, static shape, and perf
    hints, then exposes :meth:`forward` returning the model's raw outputs in
    declaration order (boxes, logits for RF-DETR).
    """

    def __init__(
        self, model_path: str | Path, device: str = "GPU", accurate: bool = False
    ) -> None:
        self.model_path = str(model_path)
        self.accurate = accurate
        self.core = ov.Core()
        available = self.core.get_available_devices()
        self.device = resolve_device(device, available)
        logger.debug(f"Available devices: {available}; using {self.device}")

        # Compiled-model cache makes repeated GPU loads fast (first compile is slow).
        # The cache dir MUST be per-device: a shared dir lets a GPU-compiled blob
        # collide with the CPU load of the same IR (an OpenVINO cache-key
        # collision), silently returning a wrong model.
        cache_dir = Path(self.model_path).parent / "cache" / self.device.replace(".", "_")
        self.core.set_property({props.cache_dir: str(cache_dir)})

        model = self.core.read_model(self.model_path)
        self._maybe_reshape(model)

        # PERFORMANCE mode (reduced precision) is the deployment-realistic default
        # and where INT8 gets its iGPU speed. But on a weight-only-INT8 IR it can
        # intermittently produce wrong outputs when the same IR was compiled on
        # another device earlier in the process — an OpenVINO heisenbug. ACCURACY
        # mode is immune and isolates the *pure* quantization effect, so quality
        # eval uses it.
        if self.accurate:
            config = {hints.execution_mode: hints.ExecutionMode.ACCURACY}
        else:
            config = {
                hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                hints.execution_mode: hints.ExecutionMode.PERFORMANCE,
            }
        self.compiled_model = self.core.compile_model(model, self.device, config)

        in_shape = self.compiled_model.input(0).shape
        self.in_h, self.in_w = int(in_shape[2]), int(in_shape[3])
        self.outputs = list(self.compiled_model.outputs)
        logger.debug(
            f"Loaded {Path(self.model_path).name}: input {tuple(in_shape)}, "
            f"{len(self.outputs)} outputs on {self.device}"
        )

    def _maybe_reshape(self, model) -> None:
        """Pin a static batch-1 shape; the iGPU GPU plugin prefers it."""
        try:
            shape = model.input(0).partial_shape
            if shape.is_dynamic:
                from soccernet_tracking_edge.config import RFDETR_RESOLUTION

                size = RFDETR_RESOLUTION
                model.reshape([1, 3, size, size])
                logger.debug(f"Reshaped dynamic IR to static [1, 3, {size}, {size}]")
        except Exception as exc:  # noqa: BLE001 - reshape is best-effort
            logger.warning(f"Could not reshape model, using as-is: {exc}")

    def forward(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Raw forward pass. Returns model outputs in declaration order."""
        result = self.compiled_model(tensor)
        return [result[o] for o in self.outputs]
