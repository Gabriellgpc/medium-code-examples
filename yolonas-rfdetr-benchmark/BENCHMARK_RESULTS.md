# Benchmark Results

**Date:** 2026-04-17
**Dataset:** COCO-2017 Validation (500 samples)
**Confidence Threshold:** 0.05 (low for full mAP PR-curve coverage)
**Warmup Inferences:** 5
**Frameworks:** rfdetr 1.6.4, modern-yolonas (editable)

## PyTorch Results at 256x256

| Model           | Device | mAP    | Avg (ms) | FPS   |
|:----------------|:-------|-------:|---------:|------:|
| RFDETRNano 256  | cpu    | 0.4366 |     65.9 |  15.2 |
| YOLO-NAS-S 256  | cpu    | 0.3098 |     38.4 |  26.1 |
| RFDETRNano 256  | cuda   | 0.4367 |     17.4 |  57.6 |
| YOLO-NAS-S 256  | cuda   | 0.3097 |     10.6 |  94.3 |

## PyTorch Results at 384x384

| Model           | Device | mAP    | Avg (ms) | FPS   |
|:----------------|:-------|-------:|---------:|------:|
| RFDETRNano 384  | cpu    | 0.5052 |    106.4 |   9.4 |
| YOLO-NAS-S 384  | cpu    | 0.3910 |     59.2 |  16.9 |
| RFDETRNano 384  | cuda   | 0.5056 |     17.8 |  56.2 |
| YOLO-NAS-S 384  | cuda   | 0.3908 |     10.5 |  94.9 |

> YOLO-NAS inference via [modern-yolonas](https://github.com/CondadosAI/modern-yolonas). RF-DETR via [rfdetr](https://github.com/roboflow/rf-detr) v1.6.4.

## Resolution Sensitivity (256 → 384)

| Model       | mAP @256 | mAP @384 | Δ (abs) | Δ (rel) |
|:------------|---------:|---------:|--------:|--------:|
| RFDETRNano  |   0.4367 |   0.5056 |  +0.069 | +15.8%  |
| YOLO-NAS-S  |   0.3097 |   0.3908 |  +0.081 | +26.2%  |

> Both models benefit from higher resolution. YOLO-NAS-S actually gains *more* in relative terms, but still trails RF-DETR in absolute mAP at every tested resolution.

## OpenVINO Results at 256x256 (Intel Iris Xe iGPU)

| Model          | Precision | mAP    | Avg (ms) | FPS   |
|:---------------|:----------|-------:|---------:|------:|
| RFDETRNano 256 | FP32      | 0.4282 |     22.2 |  45.1 |
| RFDETRNano 256 | FP16      | 0.4282 |     22.0 |  45.5 |
| RFDETRNano 256 | INT8      | 0.3375 |     16.1 |  62.1 |
| YOLO-NAS-S 256 | FP32      | 0.3447 |      6.6 | 151.2 |
| YOLO-NAS-S 256 | FP16      | 0.3447 |      6.6 | 151.4 |
| YOLO-NAS-S 256 | INT8      | 0.3361 |      4.7 | 215.0 |

> Note: YOLO-NAS-S INT8 required excluding detection head nodes from quantization (`ignored_scope`) to avoid a shape inference issue on the OpenVINO GPU plugin. The backbone and neck are fully quantized to INT8.

### INT8 quantization impact

| Model       | FP32 mAP | INT8 mAP | Δ mAP   | FP32 FPS | INT8 FPS | Δ FPS |
|:------------|---------:|---------:|--------:|---------:|---------:|------:|
| RFDETRNano  |   0.4282 |   0.3375 |  −21.2% |     45.1 |     62.1 | +38%  |
| YOLO-NAS-S  |   0.3447 |   0.3361 |   −2.5% |    151.2 |    215.0 | +42%  |

> At INT8, the two models land within noise of each other on mAP (0.336 vs 0.338) — but YOLO-NAS runs 3.5x faster. For iGPU INT8 deployment, YOLO-NAS is the clear Pareto winner.

## Fine-Tuning Results (COCO Animals — 5 classes, 200 train / 50 val)

### Training Comparison

| Metric              | RFDETRNano                | YOLO-NAS-S        |
|:--------------------|:--------------------------|:------------------|
| Epochs              | 29/30 (early stopped)     | 30/30             |
| Training time       | **3604s**                 | **311s** (11.6x faster) |
| Peak GPU memory     | 2774 MB                   | 733 MB (3.8x less)|
| Best val mAP@50:95  | 0.470 (during training)   | —                 |

> Note: RF-DETR 1.6.4 uses a PyTorch Lightning training pipeline which adds per-epoch validation + checkpointing overhead vs the old 1.5.2 loop. The 11.6x training-time delta is the honest, measured ratio under the same hardware, dataset, and batch size.

### Inference Benchmark — Standardized FiftyOne COCO Eval (threshold=0.05)

| Model             | Device | mAP@0.50  | mAP@0.50:0.95 | Avg (ms) | FPS   |
|:------------------|:-------|---------:|--------------:|---------:|------:|
| RFDETRNano FT 256 | cuda   | **0.6068** |   **0.4574** |     16.0 |  62.3 |
| YOLO-NAS-S FT 256 | cuda   | 0.4681    | 0.3281        | **9.4**  | **106.4** |

**Accuracy gap:** RF-DETR beats YOLO-NAS by +30% relative mAP@0.50 (0.607 vs 0.468) and +39% relative mAP@0.50:0.95 (0.457 vs 0.328) after fine-tuning on the same 200-image custom dataset.

**Speed gap:** YOLO-NAS fine-tuned runs 1.7x faster at inference (106.4 FPS vs 62.3 FPS on CUDA).

## Key Takeaways (Updated)

### PyTorch (canonical)
- **RF-DETR wins on accuracy by a wide margin at both resolutions:** +41% relative mAP at 256 (0.437 vs 0.310) and +29% at 384 (0.506 vs 0.391).
- **YOLO-NAS wins on speed at both resolutions:** 1.6–1.7x faster on both CPU and CUDA.
- **Both models benefit from higher resolution — YOLO-NAS actually gains *more* in relative terms** (+26% going 256→384) than RF-DETR (+16%). The old "transformers need high resolution, CNNs don't" story does not hold here.
- **CUDA latency is nearly resolution-invariant** for both models (~17ms RF-DETR, ~10ms YOLO-NAS) — on GPU, just use the higher resolution. The story flips on CPU, where 384 costs ~1.6x more time than 256.
- **CPU ↔ CUDA mAP is identical** within noise, indicating a clean, reproducible evaluation pipeline.

### OpenVINO (Intel Iris Xe iGPU)
- **YOLO-NAS-S dominates on iGPU speed:** 151–215 FPS vs 45–62 FPS for RF-DETR — a 3.3–3.5x throughput advantage.
- **FP16 is effectively free** — both models maintain identical mAP with essentially no FPS change on the Iris Xe GPU plugin.
- **INT8 hurts RF-DETR (−21% mAP) dramatically more than YOLO-NAS (−2.5% mAP)** — architectural quantization design matters. YOLO-NAS was built with QA-RepVGG blocks specifically to minimize INT8 accuracy loss; RF-DETR was not.
- **At INT8, the two models land within noise of each other on mAP** (0.336 YOLO-NAS vs 0.338 RF-DETR). YOLO-NAS is effectively a tied-accuracy, 3.5x-faster alternative at this precision.
- **Peak throughput:** YOLO-NAS INT8 reaches **215 FPS** on the Intel Iris Xe iGPU at 256x256.

### Fine-Tuning (Small Dataset Transfer Learning)
- **RF-DETR is the clear accuracy winner after fine-tuning:** +30% mAP@0.50 (0.607 vs 0.468) and +39% mAP@0.50:0.95 (0.457 vs 0.328) on the same 200-image custom dataset.
- **YOLO-NAS dominates training efficiency:** **11.6x faster training** (311s vs 3604s) and **3.8x less GPU memory** (733 MB vs 2774 MB peak).
- **YOLO-NAS wins inference speed after fine-tuning too:** 1.7x faster (106.4 vs 62.3 FPS on CUDA).
- The training-time gap widened vs the previous methodology (was 2.9x, now 11.6x) because rfdetr 1.6.4 moved to PyTorch Lightning — this adds real per-epoch overhead.

## Hardware

| Component | Details |
|:----------|:--------|
| CPU       | 12th Gen Intel Core i7-12700H |
| iGPU      | Intel Iris Xe Graphics |
| dGPU      | NVIDIA GeForce RTX 3060 Laptop |

## Configuration

| Model          | Framework                                                        |
|:---------------|:-----------------------------------------------------------------|
| RFDETRNano     | [rfdetr](https://github.com/roboflow/rf-detr) v1.6.4            |
| YOLO-NAS-S     | [modern-yolonas](https://github.com/CondadosAI/modern-yolonas)  |
