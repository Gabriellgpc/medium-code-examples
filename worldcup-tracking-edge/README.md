# soccernet-tracking-edge

Real-time **player + ball tracking** for the 2026 World Cup, on an **Intel iGPU**:
**RF-DETR** (NMS-free detection) → **roboflow/trackers** (ByteTrack vs OC-SORT) →
**OpenVINO INT8**. The capstone that connects detection, tracking, and edge
deployment — with real MOT ground truth (SoccerNet-Tracking) so the numbers are
honest.

> Companion article: `ARTICLE.md`. Sibling projects: `../rf-detr-example`,
> `../yolonas-rfdetr-benchmark`, `../yolo26-rfdetr-seg-openvino`.

## Layout

```
src/soccernet_tracking_edge/
  config.py              # constants + project-root-aware, env-overridable paths
  core/
    common.py            # OpenVINO runtime (OVDetModel) + resolve_device
    rfdetr.py            # RF-DETR preprocess + sigmoid decode → sv.Detections
    tracking.py          # ByteTrack / OC-SORT factory (roboflow/trackers)
    mot.py               # MOTChallenge I/O + motmetrics evaluation
    viz.py               # tracked-box drawing + FPS badge
  cli/                   # one click command per file (the entry points)
    download.py export.py quantize.py track.py evaluate.py benchmark.py
data/ models/ output/    # artifacts (gitignored; output/ kept)
```

## Setup

```bash
uv sync                      # base install
uv sync --extra hota         # + sn-trackeval for HOTA/DetA/AssA (optional)
```

Keep large caches off a full root partition:

```bash
export WCT_DATA_DIR=$PWD/data
export HF_HOME=$PWD/data/hf
```

## Pipeline (one command per step)

```bash
# 1. Data — SoccerNet-Tracking test split (needs the NDA password; non-commercial).
export SOCCERNET_PASSWORD=...          # from https://www.soccer-net.org/data
uv run snt-download --split test

# 2. Export RF-DETR-Nano → ONNX → OpenVINO IR (fp32, fp16)
uv run snt-export

# 3. Quantize to INT8 (weight-only + full PTQ; calibrate on football frames)
uv run snt-quantize weight-only
uv run snt-quantize full --images-dir data/soccernet/.../img1 --num-samples 128

# 4. Track a sequence on the iGPU (ByteTrack or OC-SORT) → annotated mp4 + MOT preds
uv run snt-track --source data/soccernet/.../SNMOT-116 --tracker bytetrack --device GPU

# 5. Score predictions vs ground truth (MOTA / IDF1 / ID-switches)
uv run snt-evaluate --sequence data/soccernet/.../SNMOT-116 \
                    --pred output/pred_bytetrack_int8_full.txt

# 6. Speed benchmark (device × precision) → output/benchmark.md, speed.csv, latency.png
uv run snt-benchmark --devices CPU,GPU.0,GPU.1

# 7. Tracking-quality sweep: {detector}×{precision}×{tracker} → output/tracking.md
#    Soccer models need the fine-tuned IR (snt-export --variant large-deprecated
#    --weights <ckpt> --tag soccer --resolution 728) and its INT8 quantization first.
uv run snt-trackbench --sequence data/soccernet/.../SNMOT-116
```

The fine-tuned soccer detector (dedicated ball class,
[julianzu9612/RFDETR-Soccernet](https://huggingface.co/julianzu9612/RFDETR-Soccernet),
Apache-2.0) is exported with `--variant large-deprecated` (it predates the current
`RFDETRLarge` patch size) and quantized with `snt-quantize … --tag soccer`.

## Notes

- **No training.** Pretrained RF-DETR (COCO `person` + `sports ball`) is used as-is;
  the ball is where generic detection struggles — an honest finding, not a bug.
- **No ReID** in roboflow/trackers since v2.1.0 → full-occlusion crossings cause ID
  switches. Expected and reported.
- **SoccerNet terms:** non-commercial / research; do not redistribute raw video.
  Publish metrics + attributed stills only.
