"""Step 3: the control. SNet's detection head against RF-DETR, same frames, same metric.

Validation loss cannot answer this — the two models optimise different objectives,
so their losses are not on a common scale. COCO mAP is, which is why this script
exists at all.

Three things are held fixed so the comparison means something:

* **The same frames.** Both models see an identical subsample of SN-GSR valid,
  strided across all 58 sequences rather than taking a prefix.
* **The same categories.** Evaluation is restricted to player, referee and
  goalkeeper. The ball is excluded because SNet's detection head does not predict
  it — it has a dedicated full-resolution head — and leaving it in would score
  SNet as missing every ball in the dataset.
* **The same ground truth file**, scored by pycocotools rather than anything
  hand-rolled.

What is *not* held fixed, and must be said whenever these numbers are quoted:
RF-DETR-Large here is a 1280-pixel model that runs at 1.3 FPS on the target iGPU,
against SNet's 384x640 at 32.7 FPS. This measures accuracy alone. The latency
comparison already exists and points the other way, hard.
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

from soccernet_tracking_edge.core.coco_eval import evaluate_coco, summarise  # noqa: E402
from soccernet_tracking_edge.core.detect_decode import (  # noqa: E402
    boxes_to_coco_results,
    decode_detections,
    to_coco_results,
)
from soccernet_tracking_edge.core.snet_data import DET_CLASSES, SNetDataset  # noqa: E402
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402

# Categories both models emit. Ball (0) is excluded: SNet's detection head does
# not predict it.
SHARED_CATS = [1, 2, 3]
SNET_TO_GT = {v: k for k, v in DET_CLASSES.items()}      # 0->1, 1->2, 2->3
# The fine-tuned SoccerNet RF-DETR head is 0-based ball/player/referee/goalkeeper,
# which happens to match the ground-truth category ids exactly.
RFDETR_TO_GT = {0: 0, 1: 1, 2: 2, 3: 3}


@torch.no_grad()
def eval_snet(ckpt: Path, ds: SNetDataset, gt_json: Path, device: str) -> dict:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = SNetConfig(**blob["cfg"])
    model = SNetModel(cfg)
    model.load_state_dict(blob["model"])
    model.eval().to(device)

    h, w = ds.size
    scale_x, scale_y = w / 1920.0, h / 1080.0
    results, image_ids = [], []
    for i in range(len(ds)):
        sample = ds[i]
        seq, j = ds.index[i]
        rec = ds.sequences[seq][j]
        out = model(torch.from_numpy(sample["image"])[None].to(device))
        dets = decode_detections(
            out["detection"].cpu(), out["det_size"].cpu(), out["det_offset"].cpu(),
            stride=cfg.stem_stride, scale_x=scale_x, scale_y=scale_y,
        )
        results += to_coco_results(dets, [rec["id"]], SNET_TO_GT)
        image_ids.append(rec["id"])
    return evaluate_coco(gt_json, results, cat_ids=SHARED_CATS, image_ids=image_ids)


def eval_rfdetr(ds: SNetDataset, gt_json: Path, resolution: int, threshold: float) -> dict:
    """Run the fine-tuned SoccerNet RF-DETR over the same frames."""
    import cv2
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRLargeDeprecated

    from soccernet_tracking_edge.config import SOCCER_HF_REPO

    weights = hf_hub_download(SOCCER_HF_REPO, "checkpoint_best_total.pth")
    model = RFDETRLargeDeprecated(
        pretrain_weights=weights, resolution=resolution, num_classes=4,
    )
    model.optimize_for_inference()

    results, image_ids = [], []
    for i in range(len(ds)):
        seq, j = ds.index[i]
        rec = ds.sequences[seq][j]
        img = cv2.imread(str(ds.root / rec["file_name"]))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det = model.predict(rgb, threshold=threshold)
        results += boxes_to_coco_results(
            np.asarray(det.xyxy), np.asarray(det.confidence),
            np.asarray(det.class_id), rec["id"], RFDETR_TO_GT,
        )
        image_ids.append(rec["id"])
    return evaluate_coco(gt_json, results, cat_ids=SHARED_CATS, image_ids=image_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsr", default="/kaggle/working/gsr")
    ap.add_argument("--frames", default="/kaggle/tmp/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--snet-ckpt", action="append", default=[],
                    help="label=path; repeatable")
    ap.add_argument("--val-stride", type=int, default=87)
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--rfdetr-resolution", type=int, default=1280)
    ap.add_argument("--rfdetr-threshold", type=float, default=0.05)
    ap.add_argument("--skip-rfdetr", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/step3_map.json")
    args = ap.parse_args()

    gt_json = Path(args.gsr) / args.split / "detection.json"
    ds = SNetDataset(
        Path(args.frames) / args.split, gt_json,
        size=(384, 640), heads=("detection",), det_stride=4,
        augment=False, stride=args.val_stride, limit=args.val_limit,
    )
    seqs = {s for s, _ in ds.index}
    print(f"evaluating on {len(ds)} frames from {len(seqs)} sequences", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict[str, dict] = {}
    for spec in args.snet_ckpt:
        label, _, path = spec.partition("=")
        if not Path(path).exists():
            print(f"  skip {label}: {path} missing", flush=True)
            continue
        print(f"\n--- SNet: {label}", flush=True)
        report[label] = eval_snet(Path(path), ds, gt_json, device)
        print(" ", summarise(label, report[label]), flush=True)
        print("  per class:", report[label].get("per_class_AP"), flush=True)

    if not args.skip_rfdetr:
        print(f"\n--- RF-DETR (fine-tuned SoccerNet, Large @{args.rfdetr_resolution})",
              flush=True)
        try:
            report["rfdetr_soccernet"] = eval_rfdetr(
                ds, gt_json, args.rfdetr_resolution, args.rfdetr_threshold,
            )
            print(" ", summarise("rfdetr_soccernet", report["rfdetr_soccernet"]), flush=True)
            print("  per class:", report["rfdetr_soccernet"].get("per_class_AP"), flush=True)
        except Exception as exc:  # noqa: BLE001 - the run must survive a missing dep
            import traceback
            traceback.print_exc()
            report["rfdetr_soccernet"] = {"error": f"{type(exc).__name__}: {exc}"}

    Path(args.out).write_text(json.dumps({
        "split": args.split,
        "frames": len(ds),
        "sequences": len(seqs),
        "categories": SHARED_CATS,
        "note": "accuracy only; RF-DETR-Large runs at 1280px and 1.3 FPS on the "
                "target iGPU against SNet's 384x640 at 32.7 FPS",
        "results": report,
    }, indent=2))
    print(f"\nwritten to {args.out}", flush=True)


if __name__ == "__main__":
    main()
