"""M3, run against a synthetic broadcast camera instead of SoccerNet frames.

The keypoint error budget is a question about *geometry*, not about this dataset:
given a broadcast camera pose, how far does sigma pixels of keypoint noise move a
player on the pitch? That can be answered without a single frame, which is why
this exists — it produces the answer today and validates the code path that
``kaggle/measure_gsr.py`` will run on real annotations.

What it cannot do: reproduce the real distribution of camera poses, zoom levels
and how many landmarks are actually visible in a broadcast crop. The sweep below
is a plausible bracket, not a measurement of SoccerNet. Treat the numbers as an
order of magnitude to be confirmed by M3 on Kaggle.

    python scripts/keypoint_budget_synthetic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_tracking_edge.core.pitch import (  # noqa: E402
    LANDMARKS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    fit_homography,
    project,
)

FRAME_W, FRAME_H = 1920, 1080
SIGMAS_PX = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
KEYPOINT_BUDGETS = [4, 6, 8, 12, None]
TRIALS_PER_CAMERA = 20

# A bracket of main-camera setups: gantry height and setback are roughly fixed by
# the stadium, pan and zoom are what the operator moves during play.
CAMERA_SWEEP = [
    {"setback_m": 55.0, "height_m": 14.0, "look_x": lx, "focal_px": f}
    for lx in (-30.0, -15.0, 0.0, 15.0, 30.0)
    for f in (1600.0, 2400.0, 3600.0)
]


def pitch_to_image_homography(setback_m, height_m, look_x, focal_px) -> np.ndarray:
    """Homography mapping pitch metres -> image pixels for a look-at camera."""
    centre = np.array([look_x, 0.0, 0.0])
    position = np.array([0.0, -setback_m, height_m])

    forward = centre - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])          # camera: x right, y down, z fwd
    translation = -rot @ position

    intrinsics = np.array([
        [focal_px, 0.0, FRAME_W / 2.0],
        [0.0, focal_px, FRAME_H / 2.0],
        [0.0, 0.0, 1.0],
    ])
    # Points lie on the plane z = 0, so the third column of R drops out.
    return intrinsics @ np.column_stack([rot[:, 0], rot[:, 1], translation])


def sample_players(rng, n=22) -> np.ndarray:
    """Foot positions spread over the pitch, in metres."""
    return np.column_stack([
        rng.uniform(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, n),
        rng.uniform(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, n),
    ])


def main() -> None:
    names = list(LANDMARKS)
    model_pts = np.array([LANDMARKS[n] for n in names], dtype=np.float64)
    rng = np.random.default_rng(0)

    # Keyed by the *budget label*, not by k: when budget is None, k equals the
    # visible count, which collides with the fixed budgets and would silently
    # pool them together.
    errors: dict[float, dict[str, list[float]]] = {s: {} for s in SIGMAS_PX}
    visible_counts: list[int] = []
    degenerate = 0
    ill_conditioned: dict[str, int] = {}
    attempts: dict[str, int] = {}

    for cam in CAMERA_SWEEP:
        h_p2i = pitch_to_image_homography(**cam)
        h_i2p = np.linalg.inv(h_p2i)

        lm_img = project(h_p2i, model_pts)
        inside = (
            (lm_img[:, 0] >= 0) & (lm_img[:, 0] < FRAME_W)
            & (lm_img[:, 1] >= 0) & (lm_img[:, 1] < FRAME_H)
        )
        n_vis = int(inside.sum())
        visible_counts.append(n_vis)
        if n_vis < 4:
            continue
        vis_img, vis_model = lm_img[inside], model_pts[inside]

        for _ in range(TRIALS_PER_CAMERA):
            players = sample_players(rng)
            feet_img = project(h_p2i, players)
            # Only players actually in frame contribute; the rest are off-screen.
            on_screen = (
                (feet_img[:, 0] >= 0) & (feet_img[:, 0] < FRAME_W)
                & (feet_img[:, 1] >= 0) & (feet_img[:, 1] < FRAME_H)
            )
            if on_screen.sum() < 3:
                continue
            feet_img = feet_img[on_screen]
            truth = project(h_i2p, feet_img)

            for sigma in SIGMAS_PX:
                for budget in KEYPOINT_BUDGETS:
                    label = "all" if budget is None else str(budget)
                    k = n_vis if budget is None else budget
                    if k > n_vis:
                        continue
                    attempts[label] = attempts.get(label, 0) + 1
                    idx = rng.choice(n_vis, size=k, replace=False)
                    noisy = vis_img[idx] + rng.normal(0.0, sigma, size=(k, 2))
                    try:
                        # fit_homography maps image -> pitch, which is what we need.
                        h_s, residual = fit_homography(noisy, vis_model[idx])
                        got = project(h_s, feet_img)
                    except (ValueError, np.linalg.LinAlgError):
                        degenerate += 1
                        continue
                    # A fit that cannot even reproduce its own control points is
                    # degenerate geometry (three of the chosen landmarks collinear),
                    # not measurement noise. Count it separately instead of letting
                    # it masquerade as sensitivity to sigma.
                    #
                    # Known blind spot: at k=4 the DLT passes exactly through its
                    # four points whatever their configuration, so this test cannot
                    # fire there. A near-collinear 4-subset therefore shows zero
                    # residual and still yields a homography that is wrong
                    # everywhere else — which is exactly the k=4 sigma=0 column.
                    if float(np.max(residual)) > max(1.0, 4.0 * sigma):
                        ill_conditioned[label] = ill_conditioned.get(label, 0) + 1
                        continue
                    d = np.linalg.norm(got - truth, axis=1)
                    if np.isfinite(d).all():
                        errors[sigma].setdefault(label, []).extend(d.tolist())

    print(f"visible landmarks per camera: min {min(visible_counts)}, "
          f"median {int(np.median(visible_counts))}, max {max(visible_counts)} "
          f"(of {len(names)}); fits that failed outright: {degenerate}")
    print("degenerate landmark subsets (fit cannot reproduce its own points):")
    for label in [b if b is not None else "all" for b in KEYPOINT_BUDGETS]:
        lab = str(label)
        n_bad, n_try = ill_conditioned.get(lab, 0), attempts.get(lab, 0)
        if n_try:
            print(f"    k={lab:<4} {100.0 * n_bad / n_try:5.1f}%  ({n_bad}/{n_try})")
    print()

    labels = [b if b is not None else "all" for b in KEYPOINT_BUDGETS]
    header = f"{'sigma_px':>9} | " + " | ".join(f"{f'k={lb}':>16}" for lb in labels)
    print(header)
    print("-" * len(header))
    table: dict = {}
    for sigma in SIGMAS_PX:
        cells, row = [], {}
        for lb in labels:
            vals = np.asarray(errors[sigma].get(str(lb), []))
            if len(vals) == 0:
                cells.append(f"{'-':>16}")
                continue
            med, over = float(np.median(vals)), 100.0 * float((vals > 5.0).mean())
            cells.append(f"{med:8.2f}m {over:5.1f}%")
            row[str(lb)] = {
                "median_m": round(med, 3),
                "p90_m": round(float(np.percentile(vals, 90)), 3),
                "pct_over_5m": round(over, 2),
            }
        print(f"{sigma:>9} | " + " | ".join(cells))
        table[str(sigma)] = row

    print("\ncell = median player-position error, and % of players beyond GS-HOTA's 5 m")
    print("(degenerate subsets excluded from the cells and counted above)")
    out = Path(__file__).resolve().parent.parent / "output" / "keypoint_budget_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"note": "synthetic camera sweep, not SoccerNet", "cameras": len(CAMERA_SWEEP),
         "frame": [FRAME_W, FRAME_H], "table": table}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
