# Real-time or the ball: tracking World Cup football on an Intel iGPU with RF-DETR + roboflow/trackers

> Canonical version (with interactive labs) on CondadosAI:
> https://condados.ai/blog/player-ball-tracking-rfdetr-trackers-openvino-edge
> Medium is the syndicated copy — import by that URL.
> Tags: #computer-vision #IntelSoftwareInnovator #openvino #object-tracking

**TL;DR — Detection is not tracking. On a laptop Intel iGPU (no NVIDIA, no cloud), a
quantized RF-DETR detects footballers at ~25 FPS and roboflow/trackers turns those boxes
into stable identities in real time. But the ball breaks the story: a generic detector
never sees the 17-pixel ball (0% recall), and the fine-tuned model that nails it (100% ball
recall) runs at ~1.3 FPS. On this hardware you choose real-time or the ball — and here's
why, measured on real SoccerNet footage.**

## The arrow: detect → associate → ship

- **Detect** — RF-DETR (NMS-free real-time detection transformer).
- **Associate** — roboflow/trackers (ByteTrack, OC-SORT); pure motion + IoU, no ReID since v2.1.0.
- **Ship** — OpenVINO INT8 on an Intel Iris Xe iGPU.

Detection gives boxes in one frame; tracking gives them an identity across frames; the edge
decides whether either happens in real time.

## Association is just IoU + a threshold + greedy matching

A tracker builds an IoU cost matrix between last frame's tracks and this frame's detections,
then matches greedily (highest IoU first, one-to-one, above a gate). IoU only knows geometry
— so when two same-colored players cross, with no appearance model, the "nearest box" can be
the wrong player and the IDs swap. That's an **ID switch**, the signature failure.

- **ByteTrack** keeps low-confidence detections for a second association pass — how a faint,
  fast ball at confidence 0.3 survives.
- **OC-SORT** re-anchors its Kalman filter to the observation on re-detection and adds a
  momentum term — built for occlusion and non-linear motion.

MOT scoreboard, briefly: **MOTA is detection-dominated; IDF1/HOTA punish broken identities.**
A tracker can post a fine MOTA while quietly swapping who's who.

## Results (measured on the iGPU)

**Detector speed** (RF-DETR-Nano @384, forward-only, `output/benchmark.md`):

| precision | CPU | Intel iGPU | RTX 3060 via OpenVINO |
|:--|:--|:--|:--|
| FP32 | 473 ms · 2.1 FPS | **39 ms · 25.4 FPS** | — (compile error) |
| INT8 weight-only | 322 ms · 3.1 FPS | 48 ms · 21.0 FPS | 140 ms · 7.2 FPS |
| INT8 full PTQ | 222 ms · 4.5 FPS | **35 ms · 28.7 FPS** | 135 ms · 7.4 FPS |

**The ball problem** (real SoccerNet SNMOT-116):

| detector | input | person recall | ball recall | iGPU speed |
|:--|:--|:--|:--|:--|
| RF-DETR-Nano (COCO) | 384 | 90.8% | **0.0%** | **25 FPS** |
| RF-DETR-Large (fine-tuned) | 728 | 97.2% | **100%** | **1.3 FPS** |

**Tracking quality — ByteTrack vs OC-SORT** (SNMOT-116, 375 frames, `output/tracking.md`):

| detector | precision | tracker | MOTA | IDF1 | ID switches | misses |
|:--|:--|:--|:--|:--|:--|:--|
| Nano-COCO | FP32 | ByteTrack | 0.622 | 0.455 | 135 | 2311 |
| Nano-COCO | FP32 | OC-SORT | 0.470 | 0.413 | 255 | 3403 |
| Soccer | FP32 | ByteTrack | **0.846** | 0.535 | 124 | **834** |
| Soccer | FP32 | OC-SORT | 0.791 | 0.576 | 254 | 1156 |
| Soccer | INT8 | ByteTrack | 0.838 | **0.611** | 119 | 871 |
| Soccer | INT8 | OC-SORT | 0.790 | 0.616 | 244 | 1180 |

## The honest findings

- **Real-time or the ball.** No published model gives both on this iGPU today. A fine-tuned
  *Nano/Small* soccer detector is the missing piece — the clearest "what I'd build next."
- **The detector, not the tracker, is the lever.** Nano→soccer lifts MOTA 0.62→0.85 and thirds
  the misses. Detection quality is upstream of identity — by more than expected.
- **INT8 is near-free for tracking.** Quantizing moves MOTA <0.01 and nudges IDF1 up. The
  feared "INT8 → ID-switch" propagation didn't happen.
- **A device-specific INT8 trap.** The quantized Large's DINOv2 outputs garbage on the iGPU in
  OpenVINO's `PERFORMANCE` mode (max score 0.12) but is correct in `ACCURACY` mode (0.92) and
  on CPU either way. Always eval the quantized model on the target device.
- **INT8's iGPU payoff scales with model size:** ~8% on the tiny Nano (weight-only is even
  *slower* than FP32), but a real 1.7× on the 128M Large.
- **The discrete RTX 3060 lost to the built-in iGPU** — because it ran through OpenVINO's
  Intel-tuned path, not CUDA. Read it as "don't use OpenVINO on NVIDIA."
- **ByteTrack beat OC-SORT here, including on ID switches** (124 vs 254) — its low-confidence
  recovery wins; OC-SORT's occlusion reputation didn't translate on this clip. An honest null
  result beats a hyped one.

## Reproduce it

| Parameter | Value |
|:--|:--|
| CPU / iGPU | i7-12700H (20T) / Intel Iris Xe (OpenVINO `GPU.0`) |
| dGPU | RTX 3060 Laptop via OpenVINO `GPU.1` (not CUDA) |
| RAM / OS | 31 GB / Linux 6.8 |
| Versions | openvino 2026.2, nncf 3.1, rfdetr 1.7.1, trackers 2.4.0, supervision 0.29, torch 2.12, numpy 2.4 |
| Detectors | RF-DETR-Nano (COCO) @384; RF-DETR-Large fine-tuned (julianzu9612/RFDETR-Soccernet, Apache-2.0) @728 |
| Data | SoccerNet-Tracking 2023, SNMOT-116 (25 fps, 1080p) — research NDA; metrics/frames only |
| Speed method | forward-only; 8 warmup discarded, median; pre/postprocess excluded |
| INT8 | NNCF full PTQ, `ModelType.TRANSFORMER`, 128 in-domain calibration frames |
| MOT eval | motmetrics (MOTA/MOTP/IDF1/IDSW), IoU 0.5 |

Commands: `uv sync` then `snt-download / snt-export / snt-quantize / snt-track / snt-evaluate
/ snt-benchmark / snt-trackbench`. See README. Every number matches a saved `output/` artifact.

## Limitations

One sequence, one broadcast camera; image-space only (no pitch homography); the fine-tuned
checkpoint is lightly trained (4 epochs) and the ball is its weak class under occlusion;
motion-only association degrades in dense crowding (the ReID gap is real).

## References

1. Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed.), ch. 9. Springer, 2022.
2. Zhang et al., *ByteTrack*. ECCV 2022. arXiv:2110.06864.
3. Cao et al., *Observation-Centric SORT*. CVPR 2023. arXiv:2203.14360.
4. Bewley et al., *SORT*. ICIP 2016. arXiv:1602.00763.
5. Luiten et al., *HOTA*. IJCV 2021. arXiv:2009.07736.
6. Roboflow, *RF-DETR*. github.com/roboflow/rf-detr, 2025.
7. Cioppa et al., *SoccerNet-Tracking*. CVPRW 2022. arXiv:2204.06918.
8. Intel, *OpenVINO Toolkit Documentation* (2026.2). docs.openvino.ai.
9. Roboflow, *trackers* (v2.4). trackers.roboflow.com.
