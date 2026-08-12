"""Run the four-way metres decomposition on a trained checkpoint.

    python eval_metres.py --ckpt .../best_ball.pt --gsr /kaggle/working/gsr \\
        --frames /kaggle/tmp/gsr --split valid --val-stride 87 --val-limit 500

No training. This reads an existing checkpoint and answers the question none of the
per-head metrics answer: how far off, in metres, is a player that this pipeline
puts on the minimap.
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
    ap.add_argument("--val-stride", type=int, default=87)
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--kp-threshold", type=float, default=0.15,
                    help="decode threshold for the landmark heatmaps")
    ap.add_argument("--ransac-m", type=float, default=4.0,
                    help="RANSAC inlier threshold, in METRES (destination space)")
    ap.add_argument("--out", default="/kaggle/working/metres.json")
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
        kp_stride=kp_stride, augment=False,
        stride=args.val_stride, limit=args.val_limit,
    )
    scale_x, scale_y = 640 / 1920.0, 384 / 1080.0
    print(f"{len(ds)} frames, kp_stride={kp_stride}, heads={cfg.heads}", flush=True)

    buckets: dict[str, list[float]] = {
        "gt_boxes_gt_H": [], "gt_boxes_pred_H": [],
        "pred_boxes_gt_H": [], "pred_boxes_pred_H": [],
    }
    n_no_pred_h = n_no_gt_h = 0
    kp_found = []

    for i in range(len(ds)):
        sample = ds[i]
        seq, j = ds.index[i]
        rec = ds.sequences[seq][j]
        anns = ds.anns.get(rec["id"], [])

        gt_pairs = [
            a for a in anns
            if a["category_id"] in DET_CLASSES and a.get("pitch_xy")
            and a["pitch_xy"][0] is not None
        ]
        if not gt_pairs:
            continue
        h_gt = gt_homography(anns)
        if h_gt is None:
            n_no_gt_h += 1
            continue

        out = model(torch.from_numpy(sample["image"])[None].to(device))

        kp_pts, kp_ok = decode_landmarks(
            torch.sigmoid(out["pitch"])[0].cpu().numpy(), kp_stride, scale_x, scale_y,
            threshold=args.kp_threshold,
        )
        kp_found.append(int(kp_ok.sum()))
        h_pred = homography_from_landmarks(kp_pts, kp_ok, ransac_m=args.ransac_m)
        if h_pred is None:
            n_no_pred_h += 1

        gt_feet = np.stack([
            [a["bbox"][0] + a["bbox"][2] / 2.0, a["bbox"][1] + a["bbox"][3]]
            for a in gt_pairs
        ])
        gt_truth = np.array([a["pitch_xy"] for a in gt_pairs], dtype=np.float64)

        buckets["gt_boxes_gt_H"] += position_errors(h_gt, gt_feet, gt_truth).tolist()
        if h_pred is not None:
            buckets["gt_boxes_pred_H"] += position_errors(
                h_pred, gt_feet, gt_truth).tolist()

        dets = decode_detections(
            out["detection"].cpu(), out["det_size"].cpu(), out["det_offset"].cpu(),
            stride=cfg.stem_stride, scale_x=scale_x, scale_y=scale_y,
            score_threshold=0.3,
        )[0]
        pairs = match_boxes(dets, gt_pairs)
        if pairs:
            pred_feet = foot_points([p for p, _ in pairs])
            truth = np.array([g["pitch_xy"] for _, g in pairs], dtype=np.float64)
            buckets["pred_boxes_gt_H"] += position_errors(
                h_gt, pred_feet, truth).tolist()
            if h_pred is not None:
                buckets["pred_boxes_pred_H"] += position_errors(
                    h_pred, pred_feet, truth).tolist()

    labels = {
        "gt_boxes_gt_H": "GT boxes + GT homography (harness sanity, expect ~0)",
        "gt_boxes_pred_H": "GT boxes + predicted homography (isolates the pitch head)",
        "pred_boxes_gt_H": "predicted boxes + GT homography (isolates the detector)",
        "pred_boxes_pred_H": "the pipeline as it would ship",
    }
    report = {
        "checkpoint": args.ckpt,
        "frames": len(ds),
        "kp_stride": kp_stride,
        "kp_threshold": args.kp_threshold,
        "ransac_m": args.ransac_m,
        "landmarks_found_median": float(np.median(kp_found)) if kp_found else 0,
        "frames_without_predicted_homography": n_no_pred_h,
        "frames_without_gt_homography": n_no_gt_h,
        "results": {k: summarise(np.asarray(v), labels[k]) for k, v in buckets.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nlandmarks found per frame (median): {report['landmarks_found_median']}")
    print(f"frames with no predicted homography: {n_no_pred_h}/{len(ds)}\n")
    for k in ("gt_boxes_gt_H", "gt_boxes_pred_H", "pred_boxes_gt_H", "pred_boxes_pred_H"):
        r = report["results"][k]
        if r.get("n"):
            print(f"  {labels[k]:<58} median {r['median_m']:>6.2f} m   "
                  f"p90 {r['p90_m']:>6.2f}   >5m {r['pct_over_5m']:>5.1f}%   "
                  f"<1m {r['pct_under_1m']:>5.1f}%   n={r['n']}")
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
