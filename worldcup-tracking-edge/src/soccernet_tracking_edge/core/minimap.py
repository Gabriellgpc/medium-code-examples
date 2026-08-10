"""Tactical minimap: draw the pitch, then the players projected onto it.

Rendering is deliberately separate from the geometry in ``pitch.py`` so the
homography can be tested numerically without producing a single pixel.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccernet_tracking_edge.core import pitch

GRASS = (38, 92, 46)
LINE = (235, 235, 235)
BALL_COLOR = (255, 255, 255)


class Minimap:
    """A fixed-size top-down pitch canvas with metre -> pixel mapping.

    ``px_per_m`` sets the resolution; ``margin_m`` is the grass drawn outside the
    touchlines, so a player standing on the line is not clipped at the border.
    """

    def __init__(self, px_per_m: float = 8.0, margin_m: float = 3.0) -> None:
        self.px_per_m = float(px_per_m)
        self.margin_m = float(margin_m)
        self.width = int(round((pitch.PITCH_LENGTH + 2 * margin_m) * px_per_m))
        self.height = int(round((pitch.PITCH_WIDTH + 2 * margin_m) * px_per_m))
        self._base = self._draw_pitch()

    def to_px(self, pitch_xy: np.ndarray) -> np.ndarray:
        """Pitch metres (origin at the centre spot) -> canvas pixels."""
        pitch_xy = np.asarray(pitch_xy, dtype=np.float64).reshape(-1, 2)
        x = (pitch_xy[:, 0] + pitch.PITCH_LENGTH / 2 + self.margin_m) * self.px_per_m
        y = (pitch_xy[:, 1] + pitch.PITCH_WIDTH / 2 + self.margin_m) * self.px_per_m
        return np.stack([x, y], axis=1)

    def _p(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = self.to_px(np.array([[x_m, y_m]]))[0]
        return int(round(px[0])), int(round(px[1]))

    def _draw_pitch(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), GRASS, dtype=np.uint8)
        t = max(1, int(round(self.px_per_m / 6)))
        half_l, half_w = pitch.PITCH_LENGTH / 2, pitch.PITCH_WIDTH / 2

        cv2.rectangle(img, self._p(-half_l, -half_w), self._p(half_l, half_w), LINE, t)
        cv2.line(img, self._p(0, -half_w), self._p(0, half_w), LINE, t)
        cv2.circle(img, self._p(0, 0), int(round(pitch.CENTER_CIRCLE_R * self.px_per_m)), LINE, t)
        cv2.circle(img, self._p(0, 0), max(2, t), LINE, -1)

        for side in (-1, 1):
            gl = side * half_l  # goal line x
            for depth, half in (
                (pitch.PENALTY_AREA_DEPTH, pitch.PENALTY_AREA_HALF_W),
                (pitch.GOAL_AREA_DEPTH, pitch.GOAL_AREA_HALF_W),
            ):
                cv2.rectangle(
                    img, self._p(gl, -half), self._p(gl - side * depth, half), LINE, t
                )
            cv2.circle(
                img,
                self._p(gl - side * pitch.PENALTY_SPOT_DIST, 0),
                max(2, t),
                LINE,
                -1,
            )
        return img

    def render(
        self,
        players_xy: np.ndarray,
        colors: list[tuple[int, int, int]] | None = None,
        labels: list[str] | None = None,
        ball_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw player dots (and the ball) on a fresh copy of the pitch."""
        img = self._base.copy()
        players_xy = np.asarray(players_xy, dtype=np.float64).reshape(-1, 2)
        r = max(3, int(round(self.px_per_m * 0.55)))

        for i, px in enumerate(self.to_px(players_xy)):
            c = colors[i] if colors else (60, 160, 255)
            center = (int(round(px[0])), int(round(px[1])))
            cv2.circle(img, center, r, c, -1)
            cv2.circle(img, center, r, (20, 20, 20), 1)
            if labels and i < len(labels) and labels[i]:
                cv2.putText(
                    img, labels[i], (center[0] + r + 2, center[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 245, 245), 1, cv2.LINE_AA,
                )

        if ball_xy is not None and len(np.asarray(ball_xy).reshape(-1, 2)):
            bpx = self.to_px(ball_xy)[0]
            center = (int(round(bpx[0])), int(round(bpx[1])))
            cv2.circle(img, center, max(2, r - 2), BALL_COLOR, -1)
            cv2.circle(img, center, max(2, r - 2), (20, 20, 20), 1)
        return img


def side_by_side(frame: np.ndarray, minimap: np.ndarray, gap: int = 8) -> np.ndarray:
    """Stack the frame above the minimap, matching widths."""
    w = frame.shape[1]
    scale = w / minimap.shape[1]
    mini = cv2.resize(minimap, (w, int(round(minimap.shape[0] * scale))))
    out = np.zeros((frame.shape[0] + gap + mini.shape[0], w, 3), dtype=np.uint8)
    out[: frame.shape[0]] = frame
    out[frame.shape[0] + gap :] = mini
    return out


def inset(frame: np.ndarray, minimap: np.ndarray, width_frac: float = 0.3,
          margin: int = 16, alpha: float = 0.85) -> np.ndarray:
    """Blend the minimap into the frame's bottom-right corner."""
    out = frame.copy()
    w = int(round(frame.shape[1] * width_frac))
    h = int(round(minimap.shape[0] * w / minimap.shape[1]))
    mini = cv2.resize(minimap, (w, h))
    y0, x0 = frame.shape[0] - h - margin, frame.shape[1] - w - margin
    if y0 < 0 or x0 < 0:
        return out
    roi = out[y0 : y0 + h, x0 : x0 + w]
    out[y0 : y0 + h, x0 : x0 + w] = cv2.addWeighted(mini, alpha, roi, 1 - alpha, 0)
    cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), (230, 230, 230), 1)
    return out
