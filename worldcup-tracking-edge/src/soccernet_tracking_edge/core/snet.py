"""The multi-task inference surface: one backbone, several heads, one call.

    from soccernet_tracking_edge import snet

    model = snet.SNet("models/snet_int8.xml", device="GPU.0")
    out = model.predict(frame, snet.PLAYER_DETECTION | snet.BALL_TRACKING)
    out.players      # sv.Detections
    out.ball         # BallPoint | None
    out.keypoints    # raises: KEYPOINTS was not requested

Three design decisions are baked in here, each because the alternative lies to
the caller:

**The task flags choose which graph is compiled, not which outputs are read.**
An OpenVINO IR is a static graph: asking for two heads out of three and then
discarding the third saves nothing. So a flag combination maps to its own
exported IR, and asking for a combination that was not exported is an error
rather than a silent fallback to the full model. Switching combinations at
runtime means a recompile, which ``predict`` will not do behind your back.

**Ball tracking is temporal, so the predictor holds state.** A heatmap tracker
(TrackNet, WASB) reads a window of consecutive frames; a single frame cannot
produce a ball position on its own. ``predict(frame, …)`` keeps a rolling buffer
so the call stays frame-at-a-time, and ``reset()`` must be called when the video
changes — otherwise the first frames of the new clip are matched against the
tail of the old one.

**Unrequested results raise instead of returning None.** ``None`` is a real
answer for the ball (it leaves the frame ~6% of the time). Overloading it to
also mean "you did not ask" would hide a bug behind a plausible value.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntFlag, auto
from pathlib import Path

import numpy as np


class Task(IntFlag):
    """Selectable heads. Combine with ``|``."""

    PLAYER_DETECTION = auto()   # players, goalkeepers, referees as boxes
    BALL_TRACKING = auto()      # ball centre from a temporal window
    KEYPOINTS = auto()          # pitch lines / keypoints for registration
    TEAM = auto()               # team side per player box

    @property
    def temporal(self) -> bool:
        """Whether any selected head needs more than the current frame."""
        return bool(self & Task.BALL_TRACKING)


PLAYER_DETECTION = Task.PLAYER_DETECTION
BALL_TRACKING = Task.BALL_TRACKING
KEYPOINTS = Task.KEYPOINTS
TEAM = Task.TEAM
ALL = Task.PLAYER_DETECTION | Task.BALL_TRACKING | Task.KEYPOINTS | Task.TEAM

# How many consecutive frames the ball head consumes, TrackNet-style.
BALL_WINDOW = 3


@dataclass
class BallPoint:
    """Ball centre in image pixels, with the head's confidence."""

    xy: tuple[float, float]
    score: float


@dataclass
class Prediction:
    """Results for exactly the tasks that were requested."""

    requested: Task
    _players: object | None = None
    _ball: BallPoint | None = None
    _keypoints: np.ndarray | None = None
    _team: dict[int, str] | None = field(default=None)

    def _require(self, task: Task, name: str):
        if not (self.requested & task):
            raise AttributeError(
                f"{name} was not requested: pass snet.{task.name} to predict()"
            )

    @property
    def players(self):
        self._require(Task.PLAYER_DETECTION, "players")
        return self._players

    @property
    def ball(self) -> BallPoint | None:
        """The ball, or ``None`` when the head reports it is not in frame."""
        self._require(Task.BALL_TRACKING, "ball")
        return self._ball

    @property
    def keypoints(self) -> np.ndarray | None:
        self._require(Task.KEYPOINTS, "keypoints")
        return self._keypoints

    @property
    def team(self) -> dict[int, str] | None:
        self._require(Task.TEAM, "team")
        return self._team


class SNet:
    """One backbone, several heads, selected per compiled graph.

    ``tasks`` fixes the set of heads this instance can serve; it is what the
    exported IR contains. ``predict`` may ask for a subset of it (cheap, the
    graph already runs), but never for a task outside it.
    """

    def __init__(self, model_path: str | Path, tasks: Task = ALL, device: str = "GPU.0") -> None:
        self.model_path = Path(model_path)
        self.tasks = tasks
        self.device = device
        self._buffer: deque[np.ndarray] = deque(maxlen=BALL_WINDOW)
        self._compiled = None  # set by _compile(); the OpenVINO CompiledModel

    def reset(self) -> None:
        """Drop the temporal buffer. Call this between videos."""
        self._buffer.clear()

    def predict(self, frame: np.ndarray, tasks: Task | None = None) -> Prediction:
        """Run the backbone once and return the requested heads' outputs."""
        wanted = self.tasks if tasks is None else tasks
        missing = wanted & ~self.tasks
        if missing:
            raise ValueError(
                f"{missing!r} is not in this model's exported heads ({self.tasks!r}). "
                "Export an IR for that combination instead of switching at runtime."
            )

        if wanted.temporal:
            self._buffer.append(frame)
            if len(self._buffer) < BALL_WINDOW:
                # Not enough history yet: everything else still answers, and the
                # ball is reported as absent rather than guessed from padding.
                return self._run(frame, wanted, ball_ready=False)
        return self._run(frame, wanted, ball_ready=True)

    def _run(self, frame: np.ndarray, wanted: Task, ball_ready: bool) -> Prediction:
        raise NotImplementedError(
            "no trained backbone yet — this is the inference contract the training "
            "targets. Heads to fill in: detection boxes, ball heatmap, pitch lines."
        )
