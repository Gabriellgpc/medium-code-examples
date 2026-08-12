"""Steps 1-2: train SNet heads, alone and jointly.

Step 1 trains each head on its own so there is a single-task reference to compare
against. This is not bookkeeping — the whole publishable question is *does one
backbone hurt*, and a joint model scoring 85 means nothing until the solo head's
number is known. Step 2 then trains jointly with uncertainty weighting.

The other comparison this script exists to run is the one Step 0 opened: the
WASB-faithful full-resolution trunk (``stem_stride=1``) against the decoder
variant (``stem_stride=4, head_upsample=4``) that measured 7x faster on the iGPU
for the same output resolution. Latency already favours the decoder decisively;
what is unknown is the accuracy it costs.

    python train_snet.py --heads ball --epochs 8
    python train_snet.py --heads ball --stem-stride 1 --head-upsample 1 --epochs 8
    python train_snet.py --heads ball,detection,pitch --epochs 12

Checkpoints and metrics go to /kaggle/working (persisted); frames stay in scratch.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

OUT = Path("/kaggle/working/snet")
GSR = Path("/kaggle/working/gsr")
SCRATCH = Path("/kaggle/tmp/gsr")


def locate_package() -> None:
    for c in Path("/kaggle/input").rglob("soccernet_tracking_edge/__init__.py"):
        sys.path.insert(0, str(c.parent.parent))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


locate_package()

from soccernet_tracking_edge.core.snet_data import SNetDataset  # noqa: E402
from soccernet_tracking_edge.core.snet_eval import (  # noqa: E402
    ball_metrics_sweep,
    keypoint_metrics,
)
from soccernet_tracking_edge.core.snet_loss import (  # noqa: E402
    UncertaintyWeighting,
    compute_losses,
)
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402
from soccernet_tracking_edge.core.targets import soft_argmax  # noqa: E402


def _rss_gb() -> float:
    """Resident memory of this process, in GB.

    A 47-channel pitch target pushed a run into the RAM OOM killer at epoch 3
    (SIGKILL, with VRAM at only 1.6 GB), which is invisible in any GPU metric.
    Recording it per epoch turns the next occurrence into a diagnosis.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def check_device() -> str:
    """Pick a device, and refuse a GPU whose architecture this torch cannot target.

    Kaggle's default free accelerator is a Tesla P100 (sm_60), and current PyTorch
    builds start at sm_70 — so ``torch.cuda.is_available()`` returns True and then
    the first convolution dies with "no kernel image is available for execution on
    the device", thirty lines deep in a stack trace, after the data has already
    been downloaded. Fail here instead, with the reason, and say what to do:
    request ``machine_shape: NvidiaTeslaT4`` (sm_75).
    """
    if not torch.cuda.is_available():
        print("no CUDA device; falling back to CPU (this will be very slow)", flush=True)
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in supported:
        raise SystemExit(
            f"{name} is sm_{major}{minor}, but this PyTorch supports {supported}.\n"
            "Re-run with machine_shape 'NvidiaTeslaT4' in kernel-metadata.json, or "
            "install a torch build that targets this architecture."
        )
    print(f"device: {name} (sm_{major}{minor})", flush=True)
    return "cuda"


def build_loaders(args, heads):
    common = dict(
        size=(args.height, args.width), heads=heads,
        ball_stride=1, det_stride=args.stem_stride, kp_stride=args.kp_stride,
        landmark_set=args.landmark_set,
    )
    train = SNetDataset(
        SCRATCH / "train", GSR / "train" / "detection.json",
        GSR / "train" / "ball_track.json", augment=True, limit=args.limit, **common,
    )
    val_root = SCRATCH / args.val_split
    val_det = GSR / args.val_split / "detection.json"
    val = SNetDataset(
        val_root, val_det, GSR / args.val_split / "ball_track.json",
        augment=False, limit=args.val_limit, stride=args.val_stride, **common,
    ) if val_det.exists() else None
    return (
        DataLoader(train, batch_size=args.batch, shuffle=True,
                   num_workers=args.workers, pin_memory=False, drop_last=True),
        DataLoader(val, batch_size=args.batch, shuffle=False,
                   num_workers=args.workers) if val else None,
    )


@torch.no_grad()
def validation_losses(model, loader, device, heads) -> dict:
    """Mean per-head loss on the validation split.

    The ball head has a real metric (WASB F1); detection and pitch do not have one
    wired up yet. Validation loss is a weaker signal, but it is the *same* signal
    for every head, which is what the single-task-versus-joint comparison needs —
    comparing a joint run's mAP against a solo run's training loss would answer
    nothing at all.
    """
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        for k, v in compute_losses(model(batch["image"]), batch, heads).items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: round(v / max(1, n), 5) for k, v in totals.items()}


@torch.no_grad()
def evaluate_ball(model, loader, device, scale_x: float, scale_y: float) -> dict:
    """Decode the ball head with centre-of-heatmap and score it WASB-style.

    Predictions are mapped back to *native* pixels before scoring, because the
    tolerance tau is defined there — scoring in network pixels would silently
    make the threshold three times looser.
    """
    model.eval()
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        heat = torch.sigmoid(model(images)["ball"]).cpu().numpy()
        for i in range(heat.shape[0]):
            got = soft_argmax(heat[i, 0], threshold=0.5)
            pred = (got[0] / scale_x, got[1] / scale_y) if got else None
            score = got[2] if got else 0.0
            gt = batch["ball_heat"][i, 0].numpy()
            truth = soft_argmax(gt, threshold=0.5)
            records.append({
                "pred": pred, "score": score,
                "truth": (truth[0] / scale_x, truth[1] / scale_y) if truth else None,
            })
    model.train()
    return ball_metrics_sweep(records)


@torch.no_grad()
def evaluate_pitch(model, loader, device, kp_stride: int, scale_x: float) -> dict:
    """Landmark localisation error in native pixels, and how many are found.

    The 12-epoch run showed why this is needed: the pitch head's validation loss
    read 6e-05 while the head was firing about two landmarks per frame out of nine.
    A loss that small on a target that is 99.95% zeros says nothing about whether
    anything was learned. This decodes predictions the way inference will —
    soft-argmax per channel — and scores them against the ground-truth peak.

    ``detected`` is the fraction of supervised landmarks the head actually emits a
    peak for; the error statistics are conditioned on those, so both numbers have
    to be read together. A head that fires once, perfectly, would show a superb
    median and a terrible detection rate.
    """
    model.eval()
    errors, n_gt, n_found = [], 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        pred = torch.sigmoid(model(images)["pitch"]).cpu().numpy()
        truth = batch["kp_heat"].numpy()
        valid = batch["kp_valid"].numpy()
        for i in range(pred.shape[0]):
            if valid[i] <= 0:
                continue
            for k in range(truth.shape[1]):
                gt = soft_argmax(truth[i, k], threshold=0.5)
                if gt is None:
                    continue
                n_gt += 1
                got = soft_argmax(pred[i, k], threshold=0.5)
                if got is None:
                    continue
                n_found += 1
                # heatmap px -> network px -> native px
                d = np.hypot(got[0] - gt[0], got[1] - gt[1]) * kp_stride / scale_x
                errors.append(d)
    model.train()
    out = keypoint_metrics(np.asarray(errors))
    out["landmarks_supervised"] = n_gt
    out["detected_fraction"] = round(n_found / n_gt, 4) if n_gt else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="ball", help="comma-separated: ball,detection,pitch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--stem-stride", type=int, default=4)
    ap.add_argument("--head-upsample", type=int, default=4)
    ap.add_argument("--kp-stride", type=int, default=4)
    # Must satisfy kp_stride == stem_stride / pitch_upsample, or the target and
    # the head land on different grids and the loss silently compares
    # mismatched resolutions.
    ap.add_argument("--pitch-upsample", type=int, default=1)
    ap.add_argument("--landmark-set", default="expanded",
                    choices=["base", "expanded"])
    ap.add_argument("--trunk-width", type=int, default=18)
    # Kept low deliberately: the dataset holds well over a million small Python
    # dicts, and each forked worker gradually materialises them through
    # copy-on-write refcount touches. Four workers with the 47-channel target
    # reached the RAM OOM killer partway through epoch 3.
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap training samples")
    ap.add_argument("--val-limit", type=int, default=1500)
    ap.add_argument("--val-stride", type=int, default=29,
                    help="subsample validation across sequences, not a prefix")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    tag = args.tag or f"{'-'.join(heads)}_s{args.stem_stride}_up{args.head_upsample}"
    run_dir = OUT / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    device = check_device()
    expected = args.stem_stride // args.pitch_upsample
    if args.kp_stride != expected:
        raise SystemExit(
            f"--kp-stride {args.kp_stride} does not match stem_stride "
            f"{args.stem_stride} / pitch_upsample {args.pitch_upsample} = {expected}; "
            "the target and the head would be on different grids"
        )
    cfg = SNetConfig(
        width=args.trunk_width, stem_stride=args.stem_stride,
        head_upsample=args.head_upsample, pitch_upsample=args.pitch_upsample,
        landmark_set=args.landmark_set, heads=heads,
    )
    model = SNetModel(cfg).to(device)

    # Kaggle's NvidiaTeslaT4 allocation is TWO T4s, and quota is billed per
    # session rather than per device, so the second GPU is free throughput.
    #
    # DataParallel rather than DDP, for a correctness reason and not only
    # simplicity: DP replicates the *forward* and gathers outputs in the main
    # process, so `compute_losses` still sees the whole batch. Every head here
    # normalises by a batch-wide positive count, and DDP — computing the loss
    # per shard — would silently change that normalisation.
    net = model
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    if n_gpu > 1:
        net = nn.DataParallel(model)
        print(f"DataParallel across {n_gpu} GPUs "
              f"(batch {args.batch} -> {args.batch // n_gpu} per GPU)", flush=True)

    weighting = UncertaintyWeighting(heads).to(device)
    params = list(model.parameters()) + list(weighting.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    train_loader, val_loader = build_loaders(args, heads)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
    )
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    print(f"run {tag} | device {device} | {model.n_params/1e6:.2f}M params", flush=True)
    print(f"train {len(train_loader.dataset)} samples, "
          f"val {len(val_loader.dataset) if val_loader else 0}", flush=True)

    scale_x, scale_y = args.width / 1920.0, args.height / 1080.0
    history = []
    best = -1.0
    best_loss: dict[str, float] = {}

    def save(path: Path) -> None:
        """Always from the base model — a DataParallel wrapper prefixes every
        key with "module." and the checkpoint stops loading into a plain
        SNetModel."""
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)

    for epoch in range(args.epochs):
        t0, running = time.time(), {}
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device, enabled=device == "cuda"):
                losses = compute_losses(net(batch["image"]), batch, heads)
                total, weights = weighting(losses)
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)
            if step % 50 == 0:
                parts = " ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
                print(f"  e{epoch} s{step}/{len(train_loader)} total={float(total):.4f} "
                      f"{parts} rss={_rss_gb():.1f}G", flush=True)

        entry = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 1),
            # Measured rather than extrapolated: batch size has been held at 8 for
            # comparability since Step 1, not because of a memory limit, and the
            # headroom was only ever estimated from a 6 GB laptop GPU.
            "rss_gb": round(_rss_gb(), 2),
            "peak_vram_gb": round(
                max(torch.cuda.max_memory_allocated(d) for d in range(n_gpu)) / 1e9, 2
            ) if n_gpu else None,
            "n_gpu": n_gpu,
            "samples_per_s": round(
                len(train_loader.dataset) / max(1e-6, time.time() - t0), 1),
            "train": {k: round(v / max(1, len(train_loader)), 5) for k, v in running.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
        }
        if val_loader and (epoch + 1) % args.eval_every == 0:
            entry["val_loss"] = validation_losses(net, val_loader, device, heads)
            print(f"  epoch {epoch}: val loss " +
                  " ".join(f"{k}={v}" for k, v in entry["val_loss"].items()), flush=True)
        if val_loader and "pitch" in heads and (epoch + 1) % args.eval_every == 0:
            entry["pitch_val"] = evaluate_pitch(
                net, val_loader, device, args.kp_stride, scale_x
            )
            print(f"  epoch {epoch}: pitch median {entry['pitch_val'].get('median_px')} px, "
                  f"detected {entry['pitch_val'].get('detected_fraction')}", flush=True)
        if val_loader and "ball" in heads and (epoch + 1) % args.eval_every == 0:
            entry["ball_val"] = evaluate_ball(net, val_loader, device, scale_x, scale_y)
            f1 = entry["ball_val"]["4"]["f1"]
            print(f"  epoch {epoch}: ball F1@4px = {f1:.4f} "
                  f"(AP {entry['ball_val']['4']['ap']:.4f})", flush=True)
            if f1 > best:
                best = f1
                save(run_dir / "best_ball.pt")

        # One checkpoint per head, selected on that head's own signal.
        #
        # A single best.pt chosen by ball F1, with detection mAP then read out of
        # it, produced a comparison that looked like a training result and was not:
        # a 6-epoch run peaked on ball F1 at epoch 3, mid-OneCycle with the
        # learning rate still high, and its detection score was measured from that
        # half-annealed model. `last.pt` is the fully annealed one and is what
        # cross-run detection comparisons should use.
        if "val_loss" in entry:
            for head, value in entry["val_loss"].items():
                if head == "ball":
                    continue
                if value < best_loss.get(head, float("inf")):
                    best_loss[head] = value
                    save(run_dir / f"best_{head}.pt")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        history.append(entry)
        (run_dir / "history.json").write_text(json.dumps(
            {"args": vars(args), "params_m": round(model.n_params / 1e6, 3),
             "history": history}, indent=2))
        print(f"epoch {epoch} done in {entry['seconds']}s", flush=True)

    save(run_dir / "last.pt")
    print(f"\nbest ball F1@4px: {best:.4f}" if best >= 0 else "\nno ball eval run")
    print(f"artifacts in {run_dir}", flush=True)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
