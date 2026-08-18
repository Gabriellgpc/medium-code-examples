"""Latency of the shipped SNet config on the target iGPU, at 47 landmarks.

    uv run python scripts/latency_final.py --ckpt path/to/last.pt

`scripts/arch_latency_sweep.py` chose the architecture back when the pitch head had
**33** output channels, and TRAINING-DESIGN §10 still quotes 25.1 ms / 39.9 FPS for
w32 from that sweep. §9.18 then settled on the **47**-point set, which widens the
pitch head's final convolution by 14 channels at 96x160. That number is therefore
stale, and an article that quotes it would be quoting a configuration nobody runs.

So this measures the thing that ships, and measures the 33-point head beside it so
the cost of the accuracy win is a number rather than an assumption. Latency does not
depend on weight *values*, but the checkpoint is loaded anyway when given: it makes
the measured graph provably the trained one rather than a config that resembles it.

Reported for both devices the article cares about, at FP32 and FP16, median of N
timed runs after warmup. **Excluded from the timing**: JPEG decode, resize,
normalisation, and all decoding (peak finding, homography fit) -- this is the
network forward pass only, which is the same boundary `arch_latency_sweep.py` used,
so the numbers are comparable to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soccernet_tracking_edge.core.pitch import get_landmarks  # noqa: E402
from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402


def build(cfg: SNetConfig, state: dict | None) -> SNetModel:
    model = SNetModel(cfg)
    if state is not None:
        model.load_state_dict(state)
    return model.eval()


def to_ir(model: SNetModel, shape: tuple[int, ...], out_dir: Path,
          fp16: bool) -> Path:
    """PyTorch -> OpenVINO IR, the same path the deployment uses."""
    example = torch.zeros(*shape)
    ov_model = ov.convert_model(model, example_input=example)
    path = out_dir / ("snet_fp16.xml" if fp16 else "snet_fp32.xml")
    ov.save_model(ov_model, path, compress_to_fp16=fp16)
    return path


def benchmark(compiled, shape, runs: int, warmup: int) -> dict:
    req = compiled.create_infer_request()
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        req.infer({0: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        req.infer({0: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = statistics.median(times)
    return {
        "median_ms": round(med, 2),
        "p10_ms": round(times[int(0.10 * len(times))], 2),
        "p90_ms": round(times[int(0.90 * len(times))], 2),
        "fps": round(1000.0 / med, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="converged checkpoint; its cfg fixes the shipped arm")
    ap.add_argument("--devices", default="GPU.0,CPU")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="output/latency_final.json")
    ap.add_argument("--rounds", type=int, default=4,
                    help="repeat the whole sweep N times, alternating arm order, "
                         "and report median plus round-to-round spread")
    ap.add_argument("--reverse", action="store_true",
                    help="measure the arms in the opposite order. The first run of "
                         "this script reported the 47-point head as 31%% FASTER than "
                         "the 33-point one, which has no mechanism -- more output "
                         "channels cannot cost less. Running both orders separates "
                         "an architecture effect from a drift-over-time effect: a "
                         "real one keeps its sign, an ordering artefact flips.")
    args = ap.parse_args()

    core = ov.Core()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    names = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}
    for d, n in names.items():
        print(f"{d}: {n}", flush=True)

    state = None
    base_cfg_kwargs: dict = {}
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        base_cfg_kwargs = dict(blob["cfg"])
        state = blob["model"]
        print(f"checkpoint: {args.ckpt} ({base_cfg_kwargs['landmark_set']}, "
              f"w{base_cfg_kwargs['width']})", flush=True)

    arms = []
    order = ("base", "expanded") if args.reverse else ("expanded", "base")
    for landmark_set in order:
        kwargs = dict(base_cfg_kwargs) if base_cfg_kwargs else dict(
            width=32, heads=("ball", "detection", "pitch"))
        kwargs["landmark_set"] = landmark_set
        cfg = SNetConfig(**kwargs)
        # Only the arm the checkpoint was trained for can load those weights; the
        # other differs in the pitch head's channel count by construction.
        use_state = state if landmark_set == base_cfg_kwargs.get(
            "landmark_set") else None
        arms.append((landmark_set, cfg, use_state))

    shape = (1, 3 * 3, 384, 640)
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="snet_ir_"))
    try:
        for rnd in range(args.rounds):
            # Alternate the arm order every round. A single ordering conflates the
            # architecture with whatever drifts over the life of the process --
            # the first version of this script measured the 47-point head as 31%
            # FASTER than the 33-point one for exactly that reason.
            round_arms = arms if rnd % 2 == 0 else list(reversed(arms))
            print(f"round {rnd + 1}/{args.rounds} "
                  f"({' then '.join(a[0] for a in round_arms)})", flush=True)
            for landmark_set, cfg, use_state in round_arms:
                model = build(cfg, use_state)
                n_kp = len(get_landmarks(landmark_set))
                for fp16 in (False, True):
                    ir = to_ir(model, shape, tmp, fp16)
                    for dev in devices:
                        compiled = core.compile_model(str(ir), dev)
                        row = benchmark(compiled, shape, args.runs, args.warmup)
                        row |= {
                            "round": rnd, "landmark_set": landmark_set,
                            "n_keypoints": n_kp, "width": cfg.width,
                            "precision": "FP16" if fp16 else "FP32",
                            "device": dev, "device_name": names[dev],
                            "weights": "trained" if use_state is not None
                            else "random",
                        }
                        results.append(row)
                        print(f"  {landmark_set:9s} {n_kp:>2}kp  "
                              f"{'FP16' if fp16 else 'FP32'}  {dev:6s}  "
                              f"{row['median_ms']:>7.2f} ms", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "shape": list(shape), "runs": args.runs, "warmup_discarded": args.warmup,
        "statistic": "median of timed runs",
        "excluded_from_timing": "decode, resize, normalisation, peak finding, "
                                "homography fit -- forward pass only",
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "devices": names,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n--- across {args.rounds} rounds: median, and the round-to-round "
          f"spread that has to be smaller than any effect claimed ---")
    summary = []
    for dev in devices:
        for prec in ("FP32", "FP16"):
            cell = {}
            for arm in ("base", "expanded"):
                vals = sorted(r["median_ms"] for r in results
                              if r["device"] == dev and r["precision"] == prec
                              and r["landmark_set"] == arm)
                if vals:
                    cell[arm] = (statistics.median(vals), vals[-1] - vals[0])
            if len(cell) != 2:
                continue
            (b, b_sp), (e, e_sp) = cell["base"], cell["expanded"]
            noise = max(b_sp, e_sp)
            diff = e - b
            verdict = ("UNRESOLVED, spread exceeds the difference"
                       if abs(diff) <= noise else f"{diff:+.2f} ms")
            summary.append({"device": dev, "precision": prec,
                            "base_median_ms": round(b, 2),
                            "base_spread_ms": round(b_sp, 2),
                            "expanded_median_ms": round(e, 2),
                            "expanded_spread_ms": round(e_sp, 2),
                            "difference_ms": round(diff, 2),
                            "resolvable": abs(diff) > noise})
            print(f"  {dev:6s} {prec}: 33kp {b:>7.2f} (spread {b_sp:>5.2f})  "
                  f"47kp {e:>7.2f} (spread {e_sp:>5.2f})  -> {verdict}")
    report["summary"] = summary
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
