"""Localise the SNet training RAM leak, on real data, in about four minutes.

Three runs have been SIGKILLed by the OOM killer with VRAM at 1.6 GB. The per-step
RSS log (section 9.16) says growth is 1.64 MB per step, monotone from step 0, and
untouched by `gc.collect()`. A synthetic dataset with the real dataset's exact
keys, shapes and dtypes does *not* reproduce it — neither draining the loader nor a
full train step (`scripts/rss_repro.py`). So the cause needs real data or the real
Kaggle image, and this script goes and gets it.

It answers one question per column:

  RSS(parent)   — is the growth in this process at all?
  RSS(children) — or in the dataloader workers, where the JPEG decode and
                  Albumentations actually run?
  after trim    — does glibc hand the pages back when asked? malloc_trim(0)
                  reclaiming the growth means arena fragmentation, which is a
                  configuration fix (MALLOC_ARENA_MAX). Growth that survives trim
                  is a real leak and becomes a torch-version question.

Run it twice, once with MALLOC_ARENA_MAX unset and once with it set to 2; the
driver does that. If the slope collapses in the second pass, the fix is an
environment variable and the six-epoch run needs nothing else.

Deliberately not a flag on train_snet.py: this allocates and frees on the same
schedule as training but has no checkpointing, no scheduler and no evaluation, so
whatever it shows is attributable to the loop and the loader alone.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _add_package_to_path() -> None:
    """Find the installed source tree wherever Kaggle mounted the dataset.

    A hardcoded /kaggle/input/<slug>/src killed an early kernel: the mount path
    depends on the dataset slug and on how many sources are attached, so it is
    resolved by looking for the package marker instead.
    """
    here = Path(__file__).resolve().parent
    for marker in (Path("/kaggle/input"), here.parent):
        hits = sorted(marker.rglob("soccernet_tracking_edge/__init__.py"))
        if hits:
            sys.path.insert(0, str(hits[0].parent.parent))
            return
    raise SystemExit("could not locate the soccernet_tracking_edge package")


_add_package_to_path()

from soccernet_tracking_edge.core.snet_data import SNetDataset  # noqa: E402
from soccernet_tracking_edge.core.snet_loss import (  # noqa: E402
    UncertaintyWeighting,
    compute_losses,
)
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402

GSR = Path("/kaggle/working/gsr")
SCRATCH = Path("/kaggle/tmp/gsr")


def _rss_kb(pid: str | int = "self") -> int:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def children_rss_gb() -> tuple[float, int]:
    """Summed RSS of this process's children, and how many there are.

    The dataloader workers are the half of the system the parent-side measurements
    cannot see, and they are where the real per-sample work happens.
    """
    total, n = 0, 0
    try:
        kids = Path("/proc/self/task").glob("*/children")
        pids = {p for f in kids for p in f.read_text().split()}
    except OSError:
        return 0.0, 0
    for pid in pids:
        kb = _rss_kb(pid)
        if kb:
            total += kb
            n += 1
    return total / 1e6, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trunk-width", type=int, default=32)
    ap.add_argument("--landmark-set", default="expanded")
    ap.add_argument("--label", default="default")
    ap.add_argument("--data-parallel", action="store_true",
                    help="wrap in nn.DataParallel, as train_snet.py does whenever "
                         "Kaggle hands out two T4s")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    libc = ctypes.CDLL("libc.so.6")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = ("ball", "detection", "pitch")

    print(f"=== {args.label} | MALLOC_ARENA_MAX="
          f"{os.environ.get('MALLOC_ARENA_MAX', '<unset>')} | torch {torch.__version__} "
          f"| {device}", flush=True)

    ds = SNetDataset(
        SCRATCH / "train", GSR / "train" / "detection.json",
        GSR / "train" / "ball_track.json", augment=True,
        size=(384, 640), heads=heads, ball_stride=1, det_stride=4,
        kp_stride=4, landmark_set=args.landmark_set,
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=False, drop_last=True)

    cfg = SNetConfig(width=args.trunk_width, heads=heads, landmark_set=args.landmark_set)
    model = SNetModel(cfg).to(device)
    net = model
    if args.data_parallel and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)
        print(f"  DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
    elif args.data_parallel:
        print("  --data-parallel asked for but only one device; running plain",
              flush=True)
    weighting = UncertaintyWeighting(heads).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(weighting.parameters()), lr=1e-3)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    samples = []
    base = None
    for step, batch in enumerate(dl):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device, enabled=device == "cuda"):
            losses = compute_losses(net(batch["image"]), batch, heads)
            total, _ = weighting(losses)
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()
        float(total.detach())

        if step % 25 == 0:
            before = _rss_kb() / 1e6
            libc.malloc_trim(0)
            after = _rss_kb() / 1e6
            kids_gb, n_kids = children_rss_gb()
            if base is None:
                base = before
            samples.append({"step": step, "parent_gb": round(before, 3),
                            "parent_after_trim_gb": round(after, 3),
                            "children_gb": round(kids_gb, 3), "n_children": n_kids})
            print(f"  s{step:>4} parent={before:6.3f}G (trim->{after:6.3f}G, "
                  f"reclaimed {before - after:+.3f}G)  children={kids_gb:6.3f}G x{n_kids} "
                  f"  delta_parent={before - base:+.3f}G", flush=True)
        if step >= args.steps:
            break

    # Slope measured from the *third* sample, not the first. Pass A of the previous
    # run reported 0.163 MB/step and was read as a small leak; it was entirely the
    # step 0 -> 25 warm-up (allocator growing into its steady state, cuDNN picking
    # algorithms) divided across 400 steps. From step 50 the same trace is flat to
    # the millibyte. A first-to-last slope cannot tell those two apart.
    first = samples[2] if len(samples) > 3 else samples[0]
    last = samples[-1]
    span = max(1, last["step"] - first["step"])
    report = {
        "label": args.label,
        "malloc_arena_max": os.environ.get("MALLOC_ARENA_MAX"),
        "data_parallel": bool(args.data_parallel and torch.cuda.device_count() > 1),
        "n_gpu": torch.cuda.device_count(),
        "torch": torch.__version__,
        "steps": last["step"],
        "slope_measured_from_step": first["step"],
        "parent_growth_gb": round(last["parent_gb"] - first["parent_gb"], 3),
        "parent_growth_mb_per_step": round(
            (last["parent_gb"] - first["parent_gb"]) * 1e3 / span, 3),
        "children_growth_gb": round(last["children_gb"] - first["children_gb"], 3),
        "children_growth_mb_per_step": round(
            (last["children_gb"] - first["children_gb"]) * 1e3 / span, 3),
        "trim_reclaimed_at_end_gb": round(
            last["parent_gb"] - last["parent_after_trim_gb"], 3),
        "samples": samples,
    }
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "samples"},
                            indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
