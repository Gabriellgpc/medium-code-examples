"""soccernet-tracking-edge: RF-DETR + roboflow/trackers, OpenVINO INT8 on an iGPU.

Public API re-exports so notebooks/tests can ``from soccernet_tracking_edge import
RFDETRDetector, make_tracker`` without reaching into submodules.

**Re-exports are lazy.** Importing this package eagerly used to pull in OpenVINO,
RF-DETR and the tracker stack, which meant a pure data-preparation job — building
COCO annotations on a Kaggle CPU box, say — could not import
``soccernet_tracking_edge.core.gamestate`` without the whole inference runtime
installed. Python imports the parent package before any submodule, so there was no
way around it from the caller's side.

The trade-off: a missing dependency now surfaces on first *attribute access*
rather than at import. That is the price of letting the light half of this package
stand on its own.
"""

from typing import TYPE_CHECKING

# attribute -> module that defines it
_EXPORTS = {
    "OVDetModel": "soccernet_tracking_edge.core.common",
    "resolve_device": "soccernet_tracking_edge.core.common",
    "RFDETRDetector": "soccernet_tracking_edge.core.rfdetr",
    "decode": "soccernet_tracking_edge.core.rfdetr",
    "preprocess": "soccernet_tracking_edge.core.rfdetr",
    "SNet": "soccernet_tracking_edge.core.snet",
    "Task": "soccernet_tracking_edge.core.snet",
    "make_tracker": "soccernet_tracking_edge.core.tracking",
}
_SUBMODULES = {"snet"}

if TYPE_CHECKING:  # keeps editors and type checkers seeing the real names
    from soccernet_tracking_edge.core import snet
    from soccernet_tracking_edge.core.common import OVDetModel, resolve_device
    from soccernet_tracking_edge.core.rfdetr import RFDETRDetector, decode, preprocess
    from soccernet_tracking_edge.core.snet import SNet, Task
    from soccernet_tracking_edge.core.tracking import make_tracker

__all__ = [
    "OVDetModel",
    "RFDETRDetector",
    "SNet",
    "Task",
    "decode",
    "make_tracker",
    "preprocess",
    "resolve_device",
    "snet",
]


def __getattr__(name: str):
    import importlib

    if name in _SUBMODULES:
        return importlib.import_module(f"soccernet_tracking_edge.core.{name}")
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
