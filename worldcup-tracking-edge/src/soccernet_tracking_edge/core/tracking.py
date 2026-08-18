"""Tracker factory over roboflow/trackers (ByteTrack, OC-SORT).

Both trackers share the same contract: ``update(sv.Detections) -> sv.Detections``
with ``.tracker_id`` populated on the *returned* object (a fresh copy — the input
is not mutated, so always use the return value). ``reset()`` clears state between
sequences. No ReID exists in the library since v2.1.0, so full-occlusion
crossings will cause ID switches — that's the honest limitation the article owns.
"""

from __future__ import annotations

from trackers import ByteTrackTracker, OCSORTTracker


def make_tracker(name: str, frame_rate: float = 25.0):
    """Construct a tracker by name at the given sequence frame rate.

    Defaults follow the library (v2.5.0). We keep them explicit so the article's
    reproducibility block and the code agree.
    """
    name = name.lower()
    if name == "bytetrack":
        # Two-stage association: high-confidence dets start/extend tracks, then a
        # second pass rescues low-confidence dets (great for a faint, fast ball).
        return ByteTrackTracker(
            lost_track_buffer=30,
            frame_rate=frame_rate,
            track_activation_threshold=0.7,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=0.6,
        )
    if name == "ocsort":
        # Observation-centric: re-updates the Kalman filter on re-detection and
        # blends a momentum/direction term, which recovers identities better
        # through occlusion and non-linear motion. No `track_activation_threshold`.
        return OCSORTTracker(
            lost_track_buffer=30,
            frame_rate=frame_rate,
            minimum_consecutive_frames=3,
            minimum_iou_threshold=0.3,
            direction_consistency_weight=0.2,
            high_conf_det_threshold=0.6,
            delta_t=3,
        )
    raise ValueError(f"unknown tracker {name!r}; expected 'bytetrack' or 'ocsort'")
