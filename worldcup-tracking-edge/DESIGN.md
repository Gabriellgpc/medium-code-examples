# DESIGN.md — soccernet-tracking-edge

*Phase-1 brainstorm & contract for the article + benchmarked project. This is the
source of truth for the build; deviate only with a note here.*

## One-line

Real-time **player + ball tracking** for the 2026 World Cup, on an **Intel iGPU**:
RF-DETR (NMS-free detection) → **roboflow/trackers** (ByteTrack vs OC-SORT) →
**OpenVINO INT8**. The capstone that connects CondadosAI's *Evaluation* cluster
(tracking/detection metrics) with its *Edge* cluster (RF-DETR + OpenVINO).

## The arrow (three stages, one pipeline)

```
detect (RF-DETR, INT8 IR)  →  associate across frames (ByteTrack / OC-SORT)  →  measure (MOT metrics) + ship (iGPU)
```

## Hook (the number the reader remembers)

The surprising, honest result — to be confirmed by the real run, candidates:
- "A generic COCO detector, quantized to INT8, tracks 22 players in real time on a
  laptop iGPU — but the **ball** is where edge detection dies, and here's the number."
- The **propagation** headline: INT8 costs ~X AP on the detector, which costs
  **+N ID-switches / −Y IDF1** downstream. *Detection quality is upstream of identity.*
- Speed: RF-DETR-Nano INT8 on the iGPU at ~Z FPS vs FP32 CPU (sequel to the ~6.6×
  Medium result).

**Rule: every number in the article matches a saved `output/` artifact. No hand-waving.**

## Reader & takeaway

Practitioners who can detect but haven't shipped *tracking* on the edge. After
reading they can: export RF-DETR → OpenVINO INT8, wire `roboflow/trackers`, and —
crucially — **read MOT metrics honestly** (why HOTA ≠ MOTA, why a faster detector
can *lower* IDF1).

## Unique contribution (vs existing CondadosAI posts — do NOT re-derive)

The **association mechanics** and the **INT8→ID-switch propagation**. The existing
`object-tracking-metrics` post defines MOTA/MOTP/IDF1/HOTA; the
`object-detection-metrics-map` post defines IoU/AP. This post **uses** them and
links out. What's new here:
1. Kalman predict → IoU **cost matrix** → Hungarian/greedy match → track lifecycle.
2. ByteTrack's two-stage (low-conf recovery) vs OC-SORT (observation-centric,
   occlusion-robust) — a real 2-way tradeoff on the same detections.
3. How quantizing the detector changes the *tracking* numbers.

## Running toy scenario (reused for every worked example)

**"Group F: Azul vs Vermelho — 3 players (A1, A2, V1) + 1 ball, 10 frames."**
Every concept computed on these exact numbers (formula → substitute → answer →
interpretation), per the repo Concrete-&-Visual HARD RULE. Concepts & where the
toy numbers appear:
- IoU association: a 3×3 cost matrix between frame *t* boxes and frame *t+1* boxes.
- ID switch: A1 and A2 cross; low IoU + no ReID → labels swap.
- ByteTrack: the ball at conf 0.3 survives the second association pass.
- MOTA/IDF1/HOTA: computed over the 10-frame toy sequence with 1 FP, 1 miss, 1 IDSW.

## Model / dataset / runtime / hardware

- **Detector:** `rfdetr` PyPI — **RF-DETR-Nano** (resolution 384). Primary path:
  **pretrained COCO** (`person` + `sports ball` → players/refs + ball; no training).
  Fallback if broadcast ball detection is near-zero: a `roboflow/sports` pretrained
  soccer RF-DETR checkpoint (dedicated ball class; still no training by us).
  Decode = **sigmoid**, 300 queries, cxcywh∈[0,1] (per `yolonas-rfdetr-benchmark`
  / seg — the `rf-detr-example` softmax path is stale). Handle gapped COCO-91 ids.
- **Trackers:** `trackers` PyPI **v2.5.0** — `ByteTrackTracker` and `OCSORTTracker`.
  `update(sv.Detections) -> sv.Detections` with `.tracker_id`; **use the return
  value** (fresh copy). `reset()` (or fresh instance) per sequence. **No ReID since
  v2.1.0** → honest limitation.
- **Runtime:** OpenVINO `>=2025.1` IR from `RFDETRNano.export()` (ONNX, opset 17) →
  `ov.convert_model` → FP32 / FP16 / **INT8** (NNCF). Static square shape from
  `resolution=`; iGPU wants static. NNCF **full PTQ** with
  `ModelType.TRANSFORMER` (activations → real iGPU speedup); weight-only as the
  "no speedup" honest contrast. Calibrate on ~128 football frames with the *same*
  numpy preprocess as inference.
- **Dataset:** **SoccerNet-Tracking** (2023 re-release), `task="tracking-2023",
  split=["test"]` (public GT, ~15–20 GB), **one sequence** (e.g. `SNMOT-116/`).
  MOTChallenge format (`gt/gt.txt` = `frame,id,x,y,w,h,conf,-1,-1,-1`), **ball is
  labeled**. NDA password (signed with `hello@condados.ai`). Terms: publish
  numbers + attributed stills/GIF; **no raw-video redistribution**; educational /
  non-commercial. Hero GIF sourced from a **CC-BY Roboflow** clip to stay
  commercial-clean.
- **MOT eval:** `sn-trackeval` (pip) for **HOTA/DetA/AssA** (matches SoccerNet
  leaderboard) + `motmetrics` (pip) cross-check for **MOTA/MOTP/IDF1/IDSW**. Both
  read MOTChallenge format → zero conversion.
- **Hardware (this laptop — no cloud, no cost):** i7-12700H (20 threads), 31 GB
  RAM, **Intel Alder Lake-P iGPU** (OpenVINO `GPU`, the hero), + NVIDIA RTX 3060
  Laptop (bonus CUDA reference column). All caches/downloads routed to the
  workspace disk (208 GB free); **keep off root** (44 GB free) via `HF_HOME` +
  project `data/`,`models/`.

## Benchmark axes (the credibility table)

- **Speed:** {FP32, INT8-weight, INT8-full} × {CPU, iGPU (GPU), CUDA} — warmup then
  timed forward-only loop → mean / p50 / p90 ms + FPS. `output/benchmark.md`,
  `speed.csv`, `latency.png`.
- **Detection quality retention:** INT8 vs FP32 (AP or a proxy on the sequence).
- **Tracking quality (net-new):** ByteTrack vs OC-SORT × {FP32 det, INT8 det} →
  **MOTA / IDF1 / HOTA / ID-switches**. This table *is* the propagation story.

## Hero visual

Annotated tracking GIF (CC-BY Roboflow clip): boxes + stable track IDs + live FPS,
ByteTrack vs OC-SORT side by side. Plus `latency.png`. (No Rerun/embedding angle
here — tracking is spatial, not embedding-space; skip Phase 5.)

## Article interactives (site repo)

- Reuse **`IoULab`** (association geometry) and **`TrackingLab`** (MOTA/IDF1/HOTA).
- **One net-new lab — "Association Lab":** drag two consecutive frames' boxes,
  move the IoU-match threshold, watch a correct match vs an **ID switch** live.
  Built the repo way on `src/lib/lab-kit.ts` (vanilla `<script>`, `<style
  is:global>`, multi-instance `data-*`). Hand-authored inline SVG per concept
  (ByteTrack low-conf recovery; OC-SORT through occlusion; the cost matrix).

## Hard-rule elements (ship-blockers)

- TL;DR + `updatedAt`; Reproducibility block (HW/OS/versions/exact CLI/runs+warmup/
  exclusions); Limitations; `## References` (≥1 book — Szeliski Ch. on tracking/
  motion; primary papers: RF-DETR, ByteTrack, OC-SORT, SORT, HOTA, SoccerNet;
  official docs: OpenVINO, NNCF, trackers, supervision). Tags MUST include
  `openvino` + `IntelSoftwareInnovator` + `computer-vision, object-tracking,
  edge-ai, rf-detr, mot, benchmark`.

## Risks / fallbacks

| Risk | Mitigation |
|---|---|
| SoccerNet NDA slow / Gmail rejected | Sign with `hello@condados.ai`; while waiting, dev against a CC-BY Roboflow clip + a tiny hand-made MOT `gt.txt` so code is ready when the password lands. |
| Generic COCO barely detects the tiny ball | Report it as an honest finding; if unusable, swap to a `roboflow/sports` pretrained soccer RF-DETR (no training). Same pipeline. |
| Non-commercial clause vs CondadosAI | Keep the post educational, non-paywalled, attributed; hero GIF from CC-BY source. |
| iGPU run-to-run drift ±10–20% | `--repeats`, median-of-runs (per seg `benchmark.py`); report p50/p90. |
| OC-SORT `track_activation_threshold` not accepted | It's ByteTrack-only; OC-SORT gate is `high_conf_det_threshold`. Don't cross-pass params. |
| INT8 silent precision loss (bad calibration) | Calibrate on in-domain football frames; report quality retention, not just speed. |

## Phase checklist

- [x] Phase 1 — research (RF-DETR/OV patterns, SoccerNet, trackers) + this DESIGN.
- [ ] Phase 2 — scaffold `soccernet_tracking_edge` (uv, src/, cli/, core/, config).
- [ ] Phase 3 — download / export / quantize / track (ByteTrack + OC-SORT).
- [ ] Phase 4 — benchmark (speed × device + MOT metrics) → `output/` artifacts.
- [ ] Phase 6 — ARTICLE.md + Association Lab + link pass / CONTENT_IDEAS.
- [ ] Phase 7 — verify on hardware, MDX post in site repo, memory, public repo.

## Project identity

- Dir: `soccernet-tracking-edge/` · package `soccernet_tracking_edge` · CLI prefix `wct`.
- Public repo (Phase 7): `CondadosAI/soccernet-tracking-edge` (allowlisted source only).
- Python 3.11; deps mirror the seg gold set (drop `ultralytics`, add `trackers`,
  `SoccerNet`, `sn-trackeval`, `motmetrics`).
