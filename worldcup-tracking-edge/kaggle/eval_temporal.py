"""Does temporal smoothing repair the homography tail?

The metres decomposition put all remaining error in the pitch head's tail: the
median frame is fine (0.76 m) and a minority collapse (p90 8.63 m, 8% beyond
GS-HOTA's tolerance after threshold tuning, plus 4% of frames producing nothing).
That profile — good almost always, occasionally catastrophic — is what temporal
filtering exists to fix, and it needs no retraining.

**This requires contiguous frames**, which the strided validation subset used
everywhere else deliberately does not have. Sampling every 87th frame destroys the
adjacency smoothing depends on, so this script walks whole sequences instead. That
also means its raw baseline is measured on a different frame set than §9.12's, and
the two are compared only within this run.

The cost of smoothing is lag on a panning camera, so the window is swept rather
than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def locate_package() -> None:
    for c in Path("/kaggle/input").rglob("soccernet_tracking_edge/__init__.py"):
        sys.path.insert(0, str(c.parent.parent))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


locate_package()

from soccernet_tracking_edge.core.detect_decode import decode_detections  # noqa: E402
from soccernet_tracking_edge.core.pitch_eval import (  # noqa: E402
    decode_landmarks,
    foot_points,
    gt_homography,
    homography_from_landmarks,
    match_boxes,
    position_errors,
    smooth_homographies,
    summarise,
)
from soccernet_tracking_edge.core.snet_data import DET_CLASSES, SNetDataset  # noqa: E402
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gsr", default="/kaggle/working/gsr")
    ap.add_argument("--frames", default="/kaggle/tmp/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--windows", default="3,5,9,15")
    ap.add_argument("--out", default="/kaggle/working/temporal.json")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SNetConfig(**blob["cfg"])
    model = SNetModel(cfg)
    model.load_state_dict(blob["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    kp_stride = cfg.stem_stride // cfg.pitch_upsample

    ds = SNetDataset(
        Path(args.frames) / args.split, Path(args.gsr) / args.split / "detection.json",
        size=(384, 640), heads=cfg.heads, det_stride=cfg.stem_stride,
        kp_stride=kp_stride, augment=False, landmark_set=cfg.landmark_set,
    )
    scale_x, scale_y = 640 / 1920.0, 384 / 1080.0
    windows = [int(w) for w in args.windows.split(",")]

    chosen = sorted(ds.sequences)[: args.sequences]
    print(f"{len(chosen)} sequences, contiguous: {chosen}", flush=True)

    # index into ds.index, grouped by sequence and kept in frame order
    by_seq: dict[str, list[int]] = {s: [] for s in chosen}
    for i, (seq, _) in enumerate(ds.index):
        if seq in by_seq:
            by_seq[seq].append(i)

    raw_err: list[float] = []
    smoothed_err: dict[int, list[float]] = {w: [] for w in windows}
    n_raw_missing = n_frames = 0

    for seq in chosen:
        idxs = by_seq[seq]
        homographies: list[np.ndarray | None] = []
        per_frame: list[tuple[np.ndarray, np.ndarray] | None] = []

        for i in idxs:
            sample = ds[i]
            s, j = ds.index[i]
            rec = ds.sequences[s][j]
            anns = ds.anns.get(rec["id"], [])
            gt_pairs = [
                a for a in anns
                if a["category_id"] in DET_CLASSES and a.get("pitch_xy")
                and a["pitch_xy"][0] is not None
            ]
            if not gt_pairs or gt_homography(anns) is None:
                homographies.append(None)
                per_frame.append(None)
                continue

            out = model(torch.from_numpy(sample["image"])[None].to(device))
            pts, ok = decode_landmarks(
                torch.sigmoid(out["pitch"])[0].cpu().numpy(), kp_stride, scale_x, scale_y
            )
            homographies.append(homography_from_landmarks(pts, ok, landmark_set=cfg.landmark_set))

            dets = decode_detections(
                out["detection"].cpu(), out["det_size"].cpu(), out["det_offset"].cpu(),
                stride=cfg.stem_stride, scale_x=scale_x, scale_y=scale_y,
                score_threshold=0.3,
            )[0]
            pairs = match_boxes(dets, gt_pairs)
            per_frame.append(
                (foot_points([p for p, _ in pairs]),
                 np.array([g["pitch_xy"] for _, g in pairs], dtype=np.float64))
                if pairs else None
            )

        n_frames += len(idxs)
        n_raw_missing += sum(h is None for h in homographies)

        def score(hs, bucket, frames=per_frame):
            for h, fp in zip(hs, frames, strict=True):
                if h is None or fp is None:
                    continue
                bucket += position_errors(h, fp[0], fp[1]).tolist()

        score(homographies, raw_err)
        for w in windows:
            score(smooth_homographies(homographies, w), smoothed_err[w])
        print(f"  {seq}: {len(idxs)} frames, "
              f"{sum(h is None for h in homographies)} without homography", flush=True)

    def block(errors, label, missing):
        r = summarise(np.asarray(errors), label)
        r["frames_without_homography_pct"] = round(100 * missing / max(1, n_frames), 1)
        # Frames with no homography place nobody, so they are a failure the error
        # statistics never see. The honest headline combines both.
        r["unusable_pct"] = round(
            100 * (1 - (1 - missing / max(1, n_frames)) * (1 - r.get("pct_over_5m", 0) / 100)), 1
        )
        return r

    report = {
        "checkpoint": args.ckpt,
        "sequences": chosen,
        "frames": n_frames,
        "raw": block(raw_err, "per-frame homography", n_raw_missing),
        "smoothed": {
            str(w): block(smoothed_err[w], f"median filter, window {w}", 0)
            for w in windows
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    hdr = f"{'setting':<26} {'median':>8} {'p90':>8} {'>5m':>7} {'no-H':>7} {'unusable':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in [("raw (per frame)", report["raw"])] + [
        (f"median filter w={w}", report["smoothed"][str(w)]) for w in windows
    ]:
        print(f"{name:<26} {r['median_m']:>7.3f}m {r['p90_m']:>7.3f} "
              f"{r['pct_over_5m']:>6.1f}% {r['frames_without_homography_pct']:>6.1f}% "
              f"{r['unusable_pct']:>8.1f}%")
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
