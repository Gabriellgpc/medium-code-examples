"""Fine-tune RF-DETR Nano on VOC-2007 subset."""

import json
import time
from pathlib import Path

import click
import torch
from loguru import logger
from rfdetr import RFDETRNano

DATASET_DIR = Path("datasets/coco_animals")
OUTPUT_DIR = Path("runs/rfdetr_nano_voc")
RESOLUTION = 256
NUM_CLASSES = 5


@click.command()
@click.option("--epochs", "-e", default=30, type=int, help="Training epochs")
@click.option("--batch-size", "-b", default=8, type=int, help="Batch size")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
def main(epochs: int, batch_size: int, lr: float) -> None:
    if not DATASET_DIR.exists():
        logger.error(f"Dataset not found at {DATASET_DIR}. Run download_dataset.py first.")
        return

    logger.info(f"Fine-tuning RFDETRNano on VOC-2007 ({epochs} epochs, bs={batch_size})")

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = RFDETRNano(num_classes=NUM_CLASSES, resolution=RESOLUTION, device="cuda")

    t0 = time.time()

    model.train(
        dataset_dir=str(DATASET_DIR),
        dataset_file="yolo",
        output_dir=str(OUTPUT_DIR),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_encoder=lr * 1.5,
        lr_scheduler="cosine",
        warmup_epochs=3,
        use_ema=True,
        multi_scale=False,
        early_stopping=True,
        early_stopping_patience=10,
        tensorboard=False,
        checkpoint_interval=10,
        amp=True,
        weight_decay=1e-4,
        num_workers=4,
        seed=42,
    )

    training_time = time.time() - t0

    # Collect GPU metrics
    peak_gpu_mb = 0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Save training summary
    summary = {
        "model": "RFDETRNano",
        "resolution": RESOLUTION,
        "num_classes": NUM_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "training_time_s": round(training_time, 1),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 1),
        "checkpoint_dir": str(OUTPUT_DIR),
    }

    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training complete in {training_time:.1f}s")
    logger.info(f"  Peak GPU memory: {peak_gpu_mb:.1f} MB")
    logger.info(f"  Checkpoint: {OUTPUT_DIR}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
