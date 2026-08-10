"""Step 0: what does the SNet trunk cost on the target iGPU, before training it?

WASB reports 55.7 FPS on a V100. The Iris Xe in this laptop is roughly an order of
magnitude slower, and our input is 1.67x their pixel count, so the architecture
could plausibly land in single-digit FPS. That is not a training problem and no
amount of training fixes it — so it gets measured first, with random weights, at
the cost of an afternoon rather than thirty GPU-hours.

The sweep varies the two knobs that actually move latency (stem stride and input
resolution) plus trunk width, and reports median latency on the same device the
published article benchmarks.

    python scripts/arch_latency_sweep.py [--quick]
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

from soccernet_tracking_edge.core.snet_model import SNetConfig, SNetModel  # noqa: E402

# (label, stem_stride, height, width_px, trunk_width, head_upsample)
#
# Three families. The first keeps the trunk at full resolution the way WASB does.
# The second cuts the trunk and accepts a coarse heatmap. The third cuts the trunk
# but pays a thin decoder inside the heads to get the resolution back — the whole
# question of this sweep is whether that third family exists at a usable latency.
CONFIGS = [
    ("wasb  384x640 s1 w18 up1", 1, 384, 640, 18, 1),
    ("wasb  288x512 s1 w18 up1", 1, 288, 512, 18, 1),
    ("coarse 384x640 s2 w18 up1", 2, 384, 640, 18, 1),
    ("coarse 384x640 s4 w18 up1", 4, 384, 640, 18, 1),
    ("dec   384x640 s2 w18 up2", 2, 384, 640, 18, 2),
    ("dec   384x640 s4 w18 up2", 4, 384, 640, 18, 2),
    ("dec   384x640 s4 w18 up4", 4, 384, 640, 18, 4),
    ("dec   384x640 s4 w32 up4", 4, 384, 640, 32, 4),
    ("dec   288x512 s2 w18 up2", 2, 288, 512, 18, 2),
    ("dec   512x896 s4 w18 up4", 4, 512, 896, 18, 4),
]

# Native pixels per output pixel, for a 1920x1080 source. This is what decides
# whether a config can serve the keypoint budget of TRAINING-DESIGN section 6.6.
SOURCE_W = 1920

TARGET_FPS = 25.0  # what the published article achieves with RF-DETR Nano on this iGPU


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
    return {
        "median_ms": round(statistics.median(times), 2),
        "p10_ms": round(times[int(0.10 * len(times))], 2),
        "p90_ms": round(times[int(0.90 * len(times))], 2),
        "fps": round(1000.0 / statistics.median(times), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU.0", help="OpenVINO device (GPU.0 = Iris Xe)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="fewer runs, for a smoke test")
    args = ap.parse_args()
    runs, warmup = (8, 3) if args.quick else (args.runs, args.warmup)

    core = ov.Core()
    device_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    print(f"device: {args.device} -> {device_name}")
    print(f"protocol: {warmup} warmup discarded, median of {runs} runs, "
          f"batch 1, FP32 IR (the GPU plugin executes FP16 by default)\n")

    # IRs are written to a temp dir and deleted: this measures architecture cost,
    # the weights are random, and none of it is worth keeping on a disk at 88%.
    tmp = Path(tempfile.mkdtemp(prefix="snet_arch_"))
    results = []
    try:
        header = (f"{'config':<26} {'params':>7} {'out':>11} {'nat/px':>6} "
                  f"{'median':>10} {'fps':>7}")
        print(header)
        print("-" * len(header))
        for label, stride, h, w, width, up in CONFIGS:
            cfg = SNetConfig(stem_stride=stride, width=width, head_upsample=up)
            model = SNetModel(cfg).eval()
            shape = (1, 3 * cfg.in_frames, h, w)
            example = torch.randn(*shape)
            with torch.no_grad():
                out_hw = tuple(model(example)["ball"].shape[-2:])

            ov_model = ov.convert_model(model, example_input=example, input=[shape])
            xml = tmp / f"{label.replace(' ', '_')}.xml"
            ov.save_model(ov_model, xml, compress_to_fp16=True)
            compiled = core.compile_model(str(xml), args.device)

            stats = benchmark(compiled, shape, runs, warmup)
            row = {
                "config": label, "stem_stride": stride, "input": [h, w],
                "trunk_width": width, "head_upsample": up,
                "params_m": round(model.n_params / 1e6, 2),
                "output_hw": list(out_hw), **stats,
            }
            results.append(row)
            native_per_out = SOURCE_W / out_hw[1]
            row["native_px_per_output_px"] = round(native_per_out, 2)
            flag = "" if stats["fps"] >= TARGET_FPS else "  under"
            print(f"{label:<26} {row['params_m']:>6.2f}M {str(out_hw):>11} "
                  f"{native_per_out:>6.1f} {stats['median_ms']:>8.1f}ms "
                  f"{stats['fps']:>6.1f}{flag}")
            del compiled, ov_model
            xml.unlink(missing_ok=True)
            xml.with_suffix(".bin").unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = Path(__file__).resolve().parent.parent / "output" / "arch_latency_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": device_name, "target_fps": TARGET_FPS,
        "protocol": {"warmup_discarded": warmup, "runs": runs, "statistic": "median",
                     "batch": 1, "precision": "FP16 IR on GPU plugin",
                     "excluded": "preprocessing, decoding, postprocessing"},
        "note": "random weights; this measures architecture cost, not accuracy",
        "results": results,
    }, indent=2))
    print(f"\ntarget is {TARGET_FPS} FPS (what RF-DETR Nano reaches on this iGPU)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
