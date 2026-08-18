"""Sweep the ball decoding threshold on a trained checkpoint. No training.

    python ball_threshold_sweep.py --ckpt .../last.pt --gsr /kaggle/working/gsr \\
        --frames /kaggle/tmp/gsr --split valid

The ball head's metrics say the failure is not localisation. At the converged
checkpoint, **FN = 669 at every tolerance from 1 to 12 px** while FP falls 239 ->
117 and TP rises 489 -> 611. A prediction that is merely far away turns into a TP
as tau grows; an FN that never moves is a frame where the decoder returned
*nothing*. The ball is present in 1158 of 1500 validation frames and the head fires
in 42% of them.

`soft_argmax` returns None unless some pixel clears 0.5, and that 0.5 was inherited
from WASB rather than measured here. So before paying four GPU-hours for a training
change, this asks the free question: does the head already put a peak in the right
place at a lower confidence?

Inference runs **once** and stores per-frame the peak position at every candidate
threshold; the sweep is then arithmetic over those records. Section 9.12 did the
same thing for the pitch head and halved the unusable fraction without training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def locate_package() -> None:
    for c in Path("/kaggle/input").rglob("soccernet_tracking_edge/__init__.py"):
        sys.path.insert(0, str(c.parent.parent))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


locate_package()

from soccernet_tracking_edge.core.snet_data import SNetDataset  # noqa: E402
from soccernet_tracking_edge.core.snet_eval import ball_metrics  # noqa: E402
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402
from soccernet_tracking_edge.core.targets import soft_argmax  # noqa: E402

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
TAUS = (2.0, 4.0, 8.0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gsr", default="/kaggle/working/gsr")
    ap.add_argument("--frames", default="/kaggle/tmp/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--val-stride", type=int, default=29)
    ap.add_argument("--val-limit", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/kaggle/working/ball_threshold_sweep.json")
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
        Path(args.gsr) / args.split / "ball_track.json",
        size=(384, 640), heads=cfg.heads, ball_stride=1, det_stride=cfg.stem_stride,
        kp_stride=kp_stride, augment=False, landmark_set=cfg.landmark_set,
        stride=args.val_stride, limit=args.val_limit,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                         num_workers=2)
    scale_x, scale_y = 640 / 1920.0, 384 / 1080.0
    print(f"{len(ds)} frames, thresholds {THRESHOLDS}", flush=True)

    # records[threshold] gets one entry per frame, so every threshold is scored on
    # exactly the same frames and the comparison is paired.
    records: dict[float, list[dict]] = {t: [] for t in THRESHOLDS}
    n_truth = 0
    for batch in loader:
        heat = torch.sigmoid(model(batch["image"].to(device))["ball"]).cpu().numpy()
        for i in range(heat.shape[0]):
            truth = soft_argmax(batch["ball_heat"][i, 0].numpy(), threshold=0.5)
            truth_xy = (truth[0] / scale_x, truth[1] / scale_y) if truth else None
            n_truth += truth is not None
            for t in THRESHOLDS:
                got = soft_argmax(heat[i, 0], threshold=t)
                records[t].append({
                    "pred": (got[0] / scale_x, got[1] / scale_y) if got else None,
                    "score": got[2] if got else 0.0,
                    "truth": truth_xy,
                })

    report = {
        "checkpoint": args.ckpt, "split": args.split, "frames": len(ds),
        "frames_with_ball": n_truth,
        "note": "one inference pass; thresholds differ only in the decode",
        "results": {},
    }
    for tau in TAUS:
        rows = []
        for t in THRESHOLDS:
            m = ball_metrics(records[t], tau=tau)
            m["threshold"] = t
            m["fired"] = m["TP"] + m["FP"]
            rows.append(m)
        report["results"][str(tau)] = rows

    Path(args.out).write_text(json.dumps(report, indent=2))

    for tau in TAUS:
        print(f"\n=== tau = {tau:g} px "
              f"({n_truth}/{len(ds)} frames have a ball) ===")
        print(f"{'thr':>5} {'fired':>6} {'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'prec':>6} {'rec':>6} {'F1':>6}")
        best = max(report["results"][str(tau)], key=lambda r: r["f1"])
        for r in report["results"][str(tau)]:
            mark = "  <-- best F1" if r["threshold"] == best["threshold"] else ""
            print(f"{r['threshold']:>5.2f} {r['fired']:>6} {r['TP']:>5} {r['FP']:>5} "
                  f"{r['FN']:>5} {r['precision']:>6.3f} {r['recall']:>6.3f} "
                  f"{r['f1']:>6.4f}{mark}")
        cur = next(r for r in report["results"][str(tau)] if r["threshold"] == 0.50)
        if best["threshold"] != 0.50:
            print(f"  0.50 -> {best['threshold']:.2f} moves F1 "
                  f"{cur['f1']:.4f} -> {best['f1']:.4f} "
                  f"({(best['f1'] / max(1e-9, cur['f1']) - 1) * 100:+.1f}%)")
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
