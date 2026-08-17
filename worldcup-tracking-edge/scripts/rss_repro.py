"""Isolate the linear RSS growth seen in SNet training, without SoccerNet data.

The Kaggle log (attempt 3) shows RSS rising ~1.64 MB per step, monotonically, from
step 0 of epoch 0, with `gc.collect()` between epochs releasing nothing. The
training loop body retains nothing per step, so the suspect list is the loader
plumbing: worker IPC (shared memory), the collate, or glibc arena fragmentation.

This reproduces only the *shapes* the real dataset emits, so it runs anywhere:

  image        9 x 384 x 640  float32   8.85 MB
  ball_heat    1 x 384 x 640  float32   0.98 MB
  det_*        at stride 4              0.48 MB
  kp_heat     47 x  96 x 160  float32   2.89 MB
  ---------------------------------------------
                                       13.2 MB / sample, 105 MB / batch of 8

Stages, cheapest first — each answers one question:
  loader-only  : does simply draining the DataLoader grow RSS?
  plus-to      : does the .to(device) copy in the parent grow it?
Run with --workers 0 and 2 to separate worker IPC from everything else.
"""

from __future__ import annotations

import argparse
import ctypes
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

H, W = 384, 640
DET_S, KP_S = 4, 4
N_KP = 47


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


class ShapeOnly(Dataset):
    """Emits exactly the real dataset's keys, dtypes and shapes.

    The *content* is noise. The point is that the real per-sample cost is dominated
    by fixed-size arrays, so if the leak tracks bytes moved it will reproduce here.
    """

    def __init__(self, n: int, n_kp: int = N_KP):
        self.n = n
        self.n_kp = n_kp
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        out = {
            "image": self.rng.random((9, H, W), dtype=np.float32),
            "ball_heat": np.zeros((1, H, W), dtype=np.float32),
            "ball_visible": np.float32(1.0),
            "det_heat": np.zeros((3, H // DET_S, W // DET_S), dtype=np.float32),
            "det_size": np.zeros((2, H // DET_S, W // DET_S), dtype=np.float32),
            "det_offset": np.zeros((2, H // DET_S, W // DET_S), dtype=np.float32),
            "det_mask": np.zeros((H // DET_S, W // DET_S), dtype=np.float32),
        }
        if self.n_kp:
            out["kp_heat"] = np.zeros(
                (self.n_kp, H // KP_S, W // KP_S), dtype=np.float32)
            out["kp_present"] = np.zeros(self.n_kp, dtype=np.float32)
            out["kp_valid"] = np.float32(1.0)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-kp", type=int, default=N_KP)
    ap.add_argument("--stage", choices=["loader", "plus-to", "train"], default="loader")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trunk-width", type=int, default=32)
    ap.add_argument("--malloc-trim", action="store_true",
                    help="call glibc malloc_trim(0) at each report")
    args = ap.parse_args()

    ds = ShapeOnly(args.steps * args.batch + args.batch, n_kp=args.n_kp)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=False, drop_last=True)

    libc = ctypes.CDLL("libc.so.6") if args.malloc_trim else None
    print(f"stage={args.stage} workers={args.workers} n_kp={args.n_kp} "
          f"batch={args.batch} trim={args.malloc_trim}", flush=True)

    net = opt = scaler = weighting = None
    heads = ("ball", "detection", "pitch")
    if args.stage == "train":
        # The real thing: same model, same losses, same AMP path as train_snet.py.
        from soccernet_tracking_edge.core.snet_loss import (
            UncertaintyWeighting, compute_losses,
        )
        from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel
        cfg = SNetConfig(width=args.trunk_width, heads=heads,
                         landmark_set="expanded" if args.n_kp == 47 else "base")
        net = SNetModel(cfg).to(args.device)
        weighting = UncertaintyWeighting(heads).to(args.device)
        opt = torch.optim.AdamW(
            list(net.parameters()) + list(weighting.parameters()), lr=1e-3)
        scaler = torch.amp.GradScaler(args.device, enabled=args.device == "cuda")
        print(f"  model {net.n_params/1e6:.2f}M params on {args.device}", flush=True)

    base = None
    for step, batch in enumerate(dl):
        if args.stage == "plus-to":
            batch = {k: v.to("cpu", non_blocking=True) for k, v in batch.items()}
        elif args.stage == "train":
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(args.device, enabled=args.device == "cuda"):
                losses = compute_losses(net(batch["image"]), batch, heads)
                total, _ = weighting(losses)
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            float(total)
        if step % 50 == 0:
            if libc is not None:
                libc.malloc_trim(0)
            r = rss_gb()
            if base is None:
                base = r
            print(f"  s{step:>4} rss={r:.3f}G  delta={r - base:+.3f}G", flush=True)
        if step >= args.steps:
            break

    gc.collect()
    if libc is not None:
        libc.malloc_trim(0)
    r = rss_gb()
    print(f"  end   rss={r:.3f}G  delta={r - base:+.3f}G  "
          f"per-step={(r - base) * 1e3 / max(1, args.steps):.3f} MB", flush=True)


if __name__ == "__main__":
    main()
