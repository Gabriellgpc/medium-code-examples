"""Annotate pitch landmarks on a frame and solve for the image->pitch homography.

Click the landmarks the tool asks for, in order; it solves for ``H`` and reports
the reprojection error of every point in metres, so a bad click is visible
immediately rather than silently bending the minimap.

Scope: this produces **one** homography, which holds while the camera does not
move. Broadcast cameras pan and zoom, so a moving clip needs H re-estimated per
frame; that is the next step, and this file is the ground truth it will be
checked against.

    snt-calibrate --frame data/.../img1/000001.jpg --out output/calib.json
    snt-calibrate --from-json output/calib.json          # re-solve, no GUI
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger

from soccernet_tracking_edge.config import OUTPUT_DIR
from soccernet_tracking_edge.core import pitch

# Asked in this order: the landmarks that are easiest to see in a broadcast
# frame and spread widest across the image, which is what conditions H well.
SUGGESTED = [
    "l_pen_top_goalline", "l_pen_top_corner", "l_pen_bottom_corner", "l_pen_bottom_goalline",
    "l_goal_top_corner", "l_goal_bottom_corner", "corner_tl", "corner_bl",
    "halfway_top", "halfway_bottom", "center_spot", "l_penalty_spot",
    "r_pen_top_goalline", "r_pen_top_corner", "r_pen_bottom_corner", "r_pen_bottom_goalline",
    "corner_tr", "corner_br", "r_penalty_spot",
]


def solve_and_report(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fit H from ``[{landmark, xy}]`` and log a per-point error table."""
    img_pts = np.array([p["xy"] for p in points], dtype=np.float64)
    pitch_pts = np.array([pitch.LANDMARKS[p["landmark"]] for p in points], dtype=np.float64)
    H, errors = pitch.fit_homography(img_pts, pitch_pts)

    logger.info(f"{len(points)} correspondences → mean {errors.mean():.3f} m, "
                f"max {errors.max():.3f} m")
    # Flag against the median, not the mean: one bad click inflates the mean
    # enough to hide itself, and least squares also smears its error onto the
    # points that were placed correctly.
    cutoff = 2.0 * max(float(np.median(errors)), 0.05)
    for p, e in zip(points, errors, strict=True):
        flag = "  <-- check this one" if e > cutoff else ""
        logger.info(f"  {p['landmark']:24s} {e:6.3f} m{flag}")
    if errors.mean() > 1.0:
        logger.warning("mean error above 1 m: the annotation or the pitch model is off")
    return H, errors


def _pick(frame: np.ndarray, wanted: list[str]) -> list[dict]:
    """Interactive picker. Left-click to place, 'u' undo, 's' skip, 'q' finish."""
    points: list[dict] = []
    click_xy: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, _param):  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            click_xy.append((float(x), float(y)))

    win = "calibrate: click the landmark | u=undo  s=skip  q=finish"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, frame.shape[1]), min(900, frame.shape[0]))
    cv2.setMouseCallback(win, on_mouse)

    i = 0
    while i < len(wanted):
        canvas = frame.copy()
        for p in points:
            xy = (int(p["xy"][0]), int(p["xy"][1]))
            cv2.drawMarker(canvas, xy, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(canvas, p["landmark"], (xy[0] + 8, xy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"[{len(points)} placed]  click: {wanted[i]}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(20) & 0xFF
        if click_xy:
            points.append({"landmark": wanted[i], "xy": list(click_xy.pop())})
            i += 1
        elif key == ord("s"):
            i += 1
        elif key == ord("u") and points:
            i = max(0, i - 1)
            points.pop()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return points


@click.command("calibrate")
@click.option("--frame", type=click.Path(exists=True, path_type=Path), default=None,
              help="Frame to annotate. Omit when using --from-json.")
@click.option("--from-json", "from_json", type=click.Path(exists=True, path_type=Path),
              default=None, help="Re-solve from a saved annotation (no GUI).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Where to write the calibration (default output/calib.json).")
def calibrate(frame: Path | None, from_json: Path | None, out: Path | None) -> None:
    """Solve the image->pitch homography from clicked pitch landmarks."""
    if from_json is not None:
        data = json.loads(from_json.read_text())
        points = data["points"]
        out = out or from_json
    else:
        if frame is None:
            raise click.UsageError("pass --frame to annotate, or --from-json to re-solve")
        img = cv2.imread(str(frame))
        if img is None:
            raise click.UsageError(f"could not read {frame}")
        logger.info("click each requested landmark; 's' skips one you cannot see")
        points = _pick(img, SUGGESTED)
        if len(points) < 4:
            raise click.UsageError(f"only {len(points)} points placed; a homography needs 4")
        out = out or (OUTPUT_DIR / "calib.json")

    H, errors = solve_and_report(points)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "image": str(frame) if frame else json.loads(from_json.read_text()).get("image"),
        "points": points,
        "H_image_to_pitch": H.tolist(),
        "errors_m": errors.tolist(),
        "mean_error_m": float(errors.mean()),
        "max_error_m": float(errors.max()),
    }, indent=2))
    logger.info(f"calibration → {out}")
