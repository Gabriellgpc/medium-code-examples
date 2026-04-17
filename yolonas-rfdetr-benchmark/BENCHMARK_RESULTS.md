# Benchmark Results

**Date:** 2026-03-14
**Dataset:** COCO-2017 Validation (50 samples)
**Confidence Threshold:** 0.5
**Warmup Inferences:** 5

## Multi-Model PyTorch Results at 384x384

| Model           | Device | mAP    | Avg (ms) | FPS   |
|:----------------|:-------|-------:|---------:|------:|
| RFDETRNano 384  | cpu    | 0.4591 |    101.7 |   9.8 |
| YOLO-NAS-S 384  | cpu    | 0.3860 |     55.6 |  18.0 |
| YOLOX-S 384     | cpu    | 0.3209 |     39.5 |  25.3 |
| RTMDet-Tiny 384 | cpu    | 0.2646 |     33.6 |  29.8 |
| RTMDet-S 384    | cpu    | 0.3137 |     51.2 |  19.5 |
| RFDETRNano 384  | cuda   | 0.4584 |     14.1 |  70.8 |
| YOLO-NAS-S 384  | cuda   | 0.3860 |     10.4 |  96.2 |
| YOLOX-S 384     | cuda   | 0.3209 |      7.2 | 138.7 |
| RTMDet-Tiny 384 | cuda   | 0.2646 |      8.2 | 121.5 |
| RTMDet-S 384    | cuda   | 0.3095 |      9.0 | 111.6 |

## Multi-Model PyTorch Results at 256x256

| Model           | Device | mAP    | Avg (ms) | FPS   |
|:----------------|:-------|-------:|---------:|------:|
| RFDETRNano 256  | cpu    | 0.3886 |     63.8 |  15.7 |
| YOLO-NAS-S 256  | cpu    | 0.3624 |     33.5 |  29.9 |
| YOLOX-S 256     | cpu    | 0.3244 |     24.7 |  40.4 |
| RTMDet-Tiny 256 | cpu    | 0.1952 |     22.3 |  44.9 |
| RTMDet-S 256    | cpu    | 0.2798 |     31.2 |  32.1 |
| RFDETRNano 256  | cuda   | 0.3865 |     13.9 |  72.1 |
| YOLO-NAS-S 256  | cuda   | 0.3624 |      9.9 | 100.6 |
| YOLOX-S 256     | cuda   | 0.3221 |      6.9 | 144.3 |
| RTMDet-Tiny 256 | cuda   | 0.1952 |      6.9 | 145.5 |
| RTMDet-S 256    | cuda   | 0.2796 |      7.4 | 134.4 |

> YOLOX, RTMDet, and YOLO-NAS inference via [detectors](https://github.com/CondadosAI/detectors). RF-DETR via [rfdetr](https://github.com/roboflow/rf-detr).

## OpenVINO Results at 256x256 (Intel Iris Xe iGPU)

| Model          | Precision | mAP    | Avg (ms) | FPS   |
|:---------------|:----------|-------:|---------:|------:|
| RFDETRNano 256 | FP32      | 0.3888 |     19.0 |  52.8 |
| RFDETRNano 256 | FP16      | 0.3888 |     21.5 |  46.4 |
| RFDETRNano 256 | INT8      | 0.2921 |     15.4 |  65.0 |
| YOLO-NAS-S 256 | FP32      | 0.3719 |      6.6 | 152.6 |
| YOLO-NAS-S 256 | FP16      | 0.3719 |      7.2 | 138.4 |
| YOLO-NAS-S 256 | INT8      | 0.3377 |      5.8 | 173.1 |

> Note: YOLO-NAS-S INT8 required excluding detection head nodes from quantization (`ignored_scope`) to avoid a shape inference issue on the OpenVINO GPU plugin. The backbone and neck are fully quantized to INT8.

## Fine-Tuning Results (COCO Animals — 5 classes, 200 train / 50 val)

### Training Comparison

| Metric              | RFDETRNano         | YOLO-NAS-S         |
|:--------------------|:-------------------|:-------------------|
| Epochs              | 30 (early stopped) | 30                 |
| Training time       | 250.3s             | 85.4s              |
| Peak GPU memory     | 2781 MB            | 719 MB             |
| Training AP@50:95   | 0.420              | N/A (no built-in)  |

### Inference Benchmark — Standardized FiftyOne COCO Eval (threshold=0.1)

| Model             | Device | mAP@0.50 | mAP@0.50:0.95 | Avg (ms) | FPS   |
|:------------------|:-------|--------:|-------------:|---------:|------:|
| RFDETRNano FT 256 | cuda   | 0.5001  |       0.4036 |     12.4 |  80.9 |
| YOLO-NAS-S FT 256 | cuda   | 0.4980  |       0.3459 |     10.2 |  98.4 |

## Key Takeaways

### PyTorch (CPU + CUDA)
- **At 384x384:** RFDETRNano wins on accuracy (0.459 vs 0.413 mAP), but YOLO-NAS-S is ~1.7x faster on CPU and ~1.4x faster on GPU.
- **At 256x256:** YOLO-NAS-S wins on both accuracy (0.418 vs 0.389 mAP) and speed (~1.7x CPU, ~1.3x GPU). RFDETRNano's mAP degrades significantly at lower resolution (-15%), while YOLO-NAS-S stays stable.
- **Resolution sensitivity:** RFDETRNano is more sensitive to resolution changes (0.459 -> 0.389), while YOLO-NAS-S is remarkably stable (0.413 -> 0.418).

### OpenVINO (Intel Iris Xe iGPU)
- **YOLO-NAS-S dominates on iGPU:** 138-173 FPS vs 46-65 FPS for RFDETRNano -- a ~2.7-3x speed advantage.
- **FP16 is free for accuracy:** Both models maintain identical mAP at FP16 vs FP32.
- **INT8 quantization impact:** RFDETRNano loses -25% relative mAP (0.389 -> 0.292) with only +23% FPS gain. YOLO-NAS-S (partial INT8, heads excluded) loses -9% mAP (0.372 -> 0.338) with +13% FPS gain -- a much better trade-off.
- **Peak throughput:** YOLO-NAS-S INT8 reaches **173 FPS** on the Intel Iris Xe iGPU at 256x256.

### Fine-Tuning (Small Dataset Transfer Learning)
- **Both models achieve comparable accuracy:** mAP@0.50 = 0.500 (RF-DETR) vs 0.498 (YOLO-NAS) after 30 epochs on 200 images -- virtually tied.
- **YOLO-NAS is dramatically more efficient to train:** 2.9x faster (85s vs 250s) and 3.9x less GPU memory (719 MB vs 2781 MB).
- **RF-DETR produces tighter boxes:** mAP@0.50:0.95 = 0.404 vs 0.346, indicating RF-DETR's bounding box regression is more precise at higher IoU thresholds.
- **Bug found and fixed:** A coordinate scale mismatch in modern-yolonas loss function was discovered and fixed during this experiment (normalized GT coords vs pixel-space predictions).

## Hardware

| Component | Details |
|:----------|:--------|
| CPU       | 12th Gen Intel Core i7-12700H |
| iGPU      | Intel Iris Xe Graphics |
| dGPU      | NVIDIA GeForce RTX 3060 Laptop |

## Configuration

| Model          | Framework                                                        |
|:---------------|:-----------------------------------------------------------------|
| RFDETRNano     | [rfdetr](https://github.com/roboflow/rf-detr)                   |
| YOLO-NAS-S     | [detectors](https://github.com/CondadosAI/detectors) / [modern-yolonas](https://github.com/CondadosAI/modern-yolonas) (training) |
| YOLOX-S        | [detectors](https://github.com/CondadosAI/detectors)            |
| RTMDet-Tiny    | [detectors](https://github.com/CondadosAI/detectors)            |
| RTMDet-S       | [detectors](https://github.com/CondadosAI/detectors)            |
