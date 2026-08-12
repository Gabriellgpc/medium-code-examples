"""Pitch model and image->pitch homography.

A broadcast frame and a tactical minimap are the same plane seen two ways: the
pitch is flat, so a single 3x3 homography ``H`` maps any point on the grass to
its position on the pitch in metres. Player positions come from the *foot* point
(bottom-centre of the box), because that is the point actually touching the
plane; a box centre floats above it and lands metres away after projection.

Coordinates are in metres with the origin at the centre spot, ``+x`` toward the
right-hand goal and ``+y`` toward the bottom touchline, matching the SoccerNet
calibration convention. Dimensions follow IFAB Law 1 for a 105 x 68 m pitch.
"""

from __future__ import annotations

import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

_HALF_L = PITCH_LENGTH / 2.0   # 52.5
_HALF_W = PITCH_WIDTH / 2.0    # 34.0

CENTER_CIRCLE_R = 9.15
PENALTY_SPOT_DIST = 11.0
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_W = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_W = 9.16
GOAL_HALF_W = 3.66
CORNER_ARC_R = 1.0

# Named points a human can actually find in a frame. These are the annotation
# targets: pick four or more of them in the image and the homography follows.
LANDMARKS: dict[str, tuple[float, float]] = {
    "corner_tl": (-_HALF_L, -_HALF_W),
    "corner_tr": (_HALF_L, -_HALF_W),
    "corner_bl": (-_HALF_L, _HALF_W),
    "corner_br": (_HALF_L, _HALF_W),
    "halfway_top": (0.0, -_HALF_W),
    "halfway_bottom": (0.0, _HALF_W),
    "center_spot": (0.0, 0.0),
    "circle_top": (0.0, -CENTER_CIRCLE_R),
    "circle_bottom": (0.0, CENTER_CIRCLE_R),
    "circle_left": (-CENTER_CIRCLE_R, 0.0),
    "circle_right": (CENTER_CIRCLE_R, 0.0),
    # Left half (defending the -x goal)
    "l_penalty_spot": (-_HALF_L + PENALTY_SPOT_DIST, 0.0),
    "l_pen_top_goalline": (-_HALF_L, -PENALTY_AREA_HALF_W),
    "l_pen_bottom_goalline": (-_HALF_L, PENALTY_AREA_HALF_W),
    "l_pen_top_corner": (-_HALF_L + PENALTY_AREA_DEPTH, -PENALTY_AREA_HALF_W),
    "l_pen_bottom_corner": (-_HALF_L + PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_W),
    "l_goal_top_goalline": (-_HALF_L, -GOAL_AREA_HALF_W),
    "l_goal_bottom_goalline": (-_HALF_L, GOAL_AREA_HALF_W),
    "l_goal_top_corner": (-_HALF_L + GOAL_AREA_DEPTH, -GOAL_AREA_HALF_W),
    "l_goal_bottom_corner": (-_HALF_L + GOAL_AREA_DEPTH, GOAL_AREA_HALF_W),
    "l_post_top": (-_HALF_L, -GOAL_HALF_W),
    "l_post_bottom": (-_HALF_L, GOAL_HALF_W),
    # Right half (defending the +x goal)
    "r_penalty_spot": (_HALF_L - PENALTY_SPOT_DIST, 0.0),
    "r_pen_top_goalline": (_HALF_L, -PENALTY_AREA_HALF_W),
    "r_pen_bottom_goalline": (_HALF_L, PENALTY_AREA_HALF_W),
    "r_pen_top_corner": (_HALF_L - PENALTY_AREA_DEPTH, -PENALTY_AREA_HALF_W),
    "r_pen_bottom_corner": (_HALF_L - PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_W),
    "r_goal_top_goalline": (_HALF_L, -GOAL_AREA_HALF_W),
    "r_goal_bottom_goalline": (_HALF_L, GOAL_AREA_HALF_W),
    "r_goal_top_corner": (_HALF_L - GOAL_AREA_DEPTH, -GOAL_AREA_HALF_W),
    "r_goal_bottom_corner": (_HALF_L - GOAL_AREA_DEPTH, GOAL_AREA_HALF_W),
    "r_post_top": (_HALF_L, -GOAL_HALF_W),
    "r_post_bottom": (_HALF_L, GOAL_HALF_W),
}


def foot_point(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box: where the player meets the plane.

    ``xyxy`` is (N, 4); returns (N, 2) image points.
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)


def fit_homography(
    image_pts: np.ndarray, pitch_pts: np.ndarray, ransac_m: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography image -> pitch, with its per-point error.

    Returns ``(H, errors_m)`` where ``errors_m[i]`` is the distance **in metres**
    between the projected image point and the pitch landmark it was matched to.
    Four points is the minimum; more is better, and the error vector is what
    tells you whether the annotation is any good.

    Four is also the *dangerous* minimum, not merely the smallest that works:
    eight landmarks lie on each goal line and five on the halfway line, so 11% of
    four-point subsets contain a collinear triple, and a four-point DLT passes
    exactly through its own points however degenerate they are — the failure is
    invisible to ``errors_m``. Measured on SN-GSR, a four-point fit puts a third
    of players more than 5 m from the truth even with perfect keypoints. Prefer
    eight or more, spread across the frame.

    ``ransac_m`` switches to RANSAC with that inlier threshold, for when the
    correspondences may contain a mis-clicked point. **The unit is metres, not
    pixels**: OpenCV measures the reprojection error in the destination space,
    and the destination here is the pitch.
    """
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if len(image_pts) != len(pitch_pts):
        raise ValueError(f"{len(image_pts)} image points vs {len(pitch_pts)} pitch points")
    if len(image_pts) < 4:
        raise ValueError("a homography needs at least 4 correspondences")

    if ransac_m is None:
        H, _ = cv2.findHomography(image_pts, pitch_pts, method=0)
    else:
        H, _ = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC, ransac_m)
    if H is None:
        raise ValueError("homography fit failed: are the points collinear or duplicated?")

    projected = project(H, image_pts)
    errors = np.linalg.norm(projected - pitch_pts, axis=1)
    return H, errors


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points, returning (N, 2)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(points, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def on_pitch(pitch_pts: np.ndarray, margin: float = 5.0) -> np.ndarray:
    """Boolean mask for points inside the pitch (plus a margin, in metres).

    A projected point far outside the touchline is the signature of a bad box or
    a bad homography, so this is the cheap sanity filter before drawing.
    """
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if len(pitch_pts) == 0:
        return np.zeros(0, dtype=bool)
    return (np.abs(pitch_pts[:, 0]) <= _HALF_L + margin) & (
        np.abs(pitch_pts[:, 1]) <= _HALF_W + margin
    )


# --- expanded landmark set --------------------------------------------------
#
# Section 6.6 measured that a homography needs eight or more well-spread
# correspondences, and that the median broadcast frame shows only nine of the 33
# landmarks above. The pipeline has been operating on that boundary, and the
# residual error after threshold tuning and temporal smoothing is concentrated in
# camera positions where too few landmarks are visible at all.
#
# The principle for what gets added is *distinguishability*, not count. Points
# spaced along a plain straight line are worthless here: the network would have to
# tell the third from the fourth, and a swapped correspondence damages the fit more
# than a missing one. Circle samples are distinguishable by their angular position
# relative to the circle and the lines that cross it, which is the construction
# PnLCalib uses to get from a handful of intersections to a usable set.

_CIRCLE_SAMPLE_DEG = (30, 60, 120, 150, 210, 240, 300, 330)  # 0/90/180/270 already exist
_PENALTY_ARC_R = CENTER_CIRCLE_R


def _circle_points() -> dict[str, tuple[float, float]]:
    out = {}
    for deg in _CIRCLE_SAMPLE_DEG:
        rad = np.deg2rad(deg)
        out[f"circle_{deg:03d}"] = (
            round(CENTER_CIRCLE_R * np.cos(rad), 4),
            round(CENTER_CIRCLE_R * np.sin(rad), 4),
        )
    return out


def _penalty_arc_points() -> dict[str, tuple[float, float]]:
    """Where each penalty arc meets its penalty-area line, plus the arc's apex.

    The arc is centred on the penalty spot with the same radius as the centre
    circle; only the part outside the penalty area is painted, so the two points
    where it crosses the box line are real, visible, and unambiguous.
    """
    out = {}
    box_x = PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST          # 16.5 - 11 = 5.5
    dy = float(np.sqrt(_PENALTY_ARC_R**2 - box_x**2))       # 7.31
    for side, sign in (("l", -1.0), ("r", 1.0)):
        spot_x = sign * (_HALF_L - PENALTY_SPOT_DIST)
        line_x = sign * (_HALF_L - PENALTY_AREA_DEPTH)
        out[f"{side}_arc_top"] = (round(line_x, 4), round(-dy, 4))
        out[f"{side}_arc_bottom"] = (round(line_x, 4), round(dy, 4))
        out[f"{side}_arc_apex"] = (round(spot_x + sign * _PENALTY_ARC_R, 4), 0.0)
    return out


EXPANDED_LANDMARKS: dict[str, tuple[float, float]] = {
    **LANDMARKS, **_circle_points(), **_penalty_arc_points(),
}
