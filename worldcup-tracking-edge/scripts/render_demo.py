"""Render SNet's three heads over a video, with the minimap they feed.

    uv run python scripts/render_demo.py \
        --ckpt path/to/last.pt \
        --frames data/soccernet/tracking-2023/test/SNMOT-116/img1 \
        --out output/demo/snmot116.mp4

Everything a trained SNet produces, in one frame: athlete boxes with class and
score, the ball, the pitch landmarks it found, and the 2D minimap built from the
homography those landmarks fit. The minimap is the point of the whole model — the
boxes are only interesting because they can be placed on a pitch in metres.

**Output stays local.** The SoccerNet NDA permits publishing metrics and attributed
frames, not raw video, so a rendered sequence is for looking at, not for shipping.
`output/demo/` is gitignored for that reason.

Inference matches `eval_metres.py` exactly: same 3-frame stack (i-2, i-1, i, BGR,
resized to 640x384, /255), same decoders, same thresholds. If this looks different
from the reported numbers, one of the two is wrong.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_tracking_edge.core.detect_decode import decode_detections  # noqa: E402
from soccernet_tracking_edge.core.pitch import (  # noqa: E402
    PITCH_LENGTH,
    PITCH_WIDTH,
    foot_point,
    get_landmarks,
    project,
)
from soccernet_tracking_edge.core.pitch_eval import (  # noqa: E402
    decode_landmarks,
    homography_from_landmarks,
)
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402
from soccernet_tracking_edge.core.targets import soft_argmax  # noqa: E402

NET_W, NET_H = 640, 384

# BGR. Athletes warm, ball and geometry cool, so the two never read as one layer.
CLASS_STYLE = {
    0: ("player", (80, 200, 80)),
    1: ("referee", (60, 200, 240)),
    2: ("keeper", (200, 140, 60)),
}
BALL_COLOR = (60, 60, 255)
LANDMARK_COLOR = (255, 200, 40)
MINIMAP_LINE = (200, 200, 200)


def draw_pitch(w: int, h: int) -> np.ndarray:
    """A plain top-down pitch to draw projected positions onto."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    img[:] = (45, 75, 45)

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return (int((x_m + PITCH_LENGTH / 2) / PITCH_LENGTH * (w - 1)),
                int((y_m + PITCH_WIDTH / 2) / PITCH_WIDTH * (h - 1)))

    cv2.rectangle(img, to_px(-52.5, -34), to_px(52.5, 34), MINIMAP_LINE, 1)
    cv2.line(img, to_px(0, -34), to_px(0, 34), MINIMAP_LINE, 1)
    r = int(9.15 / PITCH_LENGTH * (w - 1))
    cv2.circle(img, to_px(0, 0), r, MINIMAP_LINE, 1)
    for sign in (-1, 1):
        cv2.rectangle(img, to_px(sign * 52.5, -20.16),
                      to_px(sign * (52.5 - 16.5), 20.16), MINIMAP_LINE, 1)
        cv2.rectangle(img, to_px(sign * 52.5, -9.16),
                      to_px(sign * (52.5 - 5.5), 9.16), MINIMAP_LINE, 1)
    return img


def pitch_to_px(pts_m: np.ndarray, w: int, h: int) -> np.ndarray:
    px = (pts_m[:, 0] + PITCH_LENGTH / 2) / PITCH_LENGTH * (w - 1)
    py = (pts_m[:, 1] + PITCH_WIDTH / 2) / PITCH_WIDTH * (h - 1)
    return np.stack([px, py], axis=1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", required=True, help="directory of numbered jpgs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--det-threshold", type=float, default=0.35)
    ap.add_argument("--ball-threshold", type=float, default=0.5)
    ap.add_argument("--kp-threshold", type=float, default=0.15)
    ap.add_argument("--ransac-m", type=float, default=4.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SNetConfig(**blob["cfg"])
    model = SNetModel(cfg)
    model.load_state_dict(blob["model"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    kp_stride = cfg.stem_stride // cfg.pitch_upsample
    landmark_names = list(get_landmarks(cfg.landmark_set))
    print(f"{cfg.landmark_set} landmarks ({len(landmark_names)}), w{cfg.width}, "
          f"{model.n_params / 1e6:.2f}M params on {device}", flush=True)

    files = sorted(Path(args.frames).glob("*.jpg"))
    if not files:
        raise SystemExit(f"no jpgs in {args.frames}")
    if args.limit:
        files = files[:args.limit]
    probe = cv2.imread(str(files[0]))
    src_h, src_w = probe.shape[:2]
    scale_x, scale_y = NET_W / src_w, NET_H / src_h

    map_w, map_h = src_w // 2, int(src_w // 2 * PITCH_WIDTH / PITCH_LENGTH)
    blank_pitch = draw_pitch(map_w, map_h)
    out_h = src_h + map_h

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (src_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")

    # Ring of the last in_frames resized frames, so each frame is decoded once
    # rather than in_frames times.
    ring: list[np.ndarray] = []
    stats = {"frames": 0, "no_h": 0, "ball": 0, "infer_s": 0.0}
    landmarks_per_frame = []

    for idx, path in enumerate(files):
        raw = cv2.imread(str(path))
        if raw is None:
            continue
        small = cv2.resize(raw, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
        if not ring:
            ring = [small] * cfg.in_frames
        else:
            ring.append(small)
            ring = ring[-cfg.in_frames:]

        stack = np.concatenate(ring, axis=2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(stack.transpose(2, 0, 1))[None].to(device)

        t0 = time.perf_counter()
        out = model(tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        stats["infer_s"] += time.perf_counter() - t0

        dets = decode_detections(
            out["detection"].cpu(), out["det_size"].cpu(), out["det_offset"].cpu(),
            stride=cfg.stem_stride, scale_x=scale_x, scale_y=scale_y,
            score_threshold=args.det_threshold,
        )[0]

        ball = soft_argmax(torch.sigmoid(out["ball"])[0, 0].cpu().numpy(),
                           threshold=args.ball_threshold)

        kp_pts, kp_ok = decode_landmarks(
            torch.sigmoid(out["pitch"])[0].cpu().numpy(), kp_stride,
            scale_x, scale_y, threshold=args.kp_threshold,
        )
        landmarks_per_frame.append(int(kp_ok.sum()))
        h_pred = homography_from_landmarks(
            kp_pts, kp_ok, ransac_m=args.ransac_m, landmark_set=cfg.landmark_set)

        canvas = raw.copy()
        for d in dets:
            x, y, bw, bh = d["bbox"]
            name, color = CLASS_STYLE.get(d["class"], ("?", (200, 200, 200)))
            p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, f"{name} {d['score']:.2f}", (p1[0], p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if ball:
            bx, by = ball[0] / scale_x, ball[1] / scale_y
            stats["ball"] += 1
            cv2.circle(canvas, (int(bx), int(by)), 11, BALL_COLOR, 2)
            cv2.drawMarker(canvas, (int(bx), int(by)), BALL_COLOR,
                           cv2.MARKER_CROSS, 22, 1)
            cv2.putText(canvas, f"ball {ball[2]:.2f}", (int(bx) + 14, int(by) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BALL_COLOR, 1, cv2.LINE_AA)

        for k in np.nonzero(kp_ok)[0]:
            px, py = int(kp_pts[k, 0]), int(kp_pts[k, 1])
            cv2.circle(canvas, (px, py), 4, LANDMARK_COLOR, -1)
            cv2.circle(canvas, (px, py), 4, (20, 20, 20), 1)

        mini = blank_pitch.copy()
        if h_pred is None:
            stats["no_h"] += 1
            cv2.putText(mini, "no homography", (10, map_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 220), 2, cv2.LINE_AA)
        else:
            boxes = np.array(
                [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]]
                 for d in dets], dtype=np.float64
            ).reshape(-1, 4)
            if len(boxes):
                on_pitch_m = project(h_pred, foot_point(boxes))
                for d, p in zip(dets, pitch_to_px(on_pitch_m, map_w, map_h),
                                strict=True):
                    _, color = CLASS_STYLE.get(d["class"], ("?", (200, 200, 200)))
                    if 0 <= p[0] < map_w and 0 <= p[1] < map_h:
                        cv2.circle(mini, (int(p[0]), int(p[1])), 5, color, -1)
                        cv2.circle(mini, (int(p[0]), int(p[1])), 5, (20, 20, 20), 1)
            if ball:
                bp = pitch_to_px(
                    project(h_pred, np.array([[ball[0] / scale_x, ball[1] / scale_y]])),
                    map_w, map_h)[0]
                if 0 <= bp[0] < map_w and 0 <= bp[1] < map_h:
                    cv2.circle(mini, (int(bp[0]), int(bp[1])), 4, BALL_COLOR, -1)

        hud = (f"frame {idx + 1}/{len(files)}   athletes {len(dets)}   "
               f"landmarks {int(kp_ok.sum())}/{len(landmark_names)}   "
               f"{'homography OK' if h_pred is not None else 'NO HOMOGRAPHY'}")
        cv2.rectangle(canvas, (0, 0), (src_w, 34), (25, 25, 25), -1)
        cv2.putText(canvas, hud, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (235, 235, 235), 1, cv2.LINE_AA)

        frame_out = np.full((out_h, src_w, 3), 25, dtype=np.uint8)
        frame_out[:src_h] = canvas
        frame_out[src_h:, :map_w] = mini

        # Legend and running counters, in the space beside the minimap.
        x0, y = map_w + 30, src_h + 34
        cv2.putText(frame_out, "minimap: foot point projected through the predicted "
                    "homography", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (225, 225, 225), 1, cv2.LINE_AA)
        y += 42
        for cls_id in sorted(CLASS_STYLE):
            name, color = CLASS_STYLE[cls_id]
            count = sum(1 for d in dets if d["class"] == cls_id)
            cv2.rectangle(frame_out, (x0, y - 12), (x0 + 22, y + 2), color, -1)
            cv2.putText(frame_out, f"{name}  {count}", (x0 + 34, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 1, cv2.LINE_AA)
            y += 32
        cv2.circle(frame_out, (x0 + 11, y - 5), 8, BALL_COLOR, 2)
        cv2.putText(frame_out, f"ball  {'found' if ball else 'not found'}",
                    (x0 + 34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (215, 215, 215), 1, cv2.LINE_AA)
        y += 32
        cv2.circle(frame_out, (x0 + 11, y - 5), 5, LANDMARK_COLOR, -1)
        cv2.putText(frame_out, f"pitch landmark  {int(kp_ok.sum())} of "
                    f"{len(landmark_names)}", (x0 + 34, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (215, 215, 215), 1, cv2.LINE_AA)
        writer.write(frame_out)

        stats["frames"] += 1
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(files)}", flush=True)

    writer.release()
    n = max(1, stats["frames"])
    print(f"\nwrote {args.out}")
    print(f"  frames                {stats['frames']}")
    print(f"  landmarks per frame   median {np.median(landmarks_per_frame):.0f}")
    print(f"  frames with no H      {stats['no_h']} ({stats['no_h'] / n * 100:.1f}%)")
    print(f"  ball found            {stats['ball']} ({stats['ball'] / n * 100:.1f}%)")
    print(f"  model time            {stats['infer_s'] / n * 1e3:.1f} ms/frame "
          f"on {device} (decode and drawing excluded)")


if __name__ == "__main__":
    main()
