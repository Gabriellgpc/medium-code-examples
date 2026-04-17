"""Fine-tune YOLO-NAS-S on VOC-2007 subset."""

import json
import time
from pathlib import Path

import click
import torch
from loguru import logger
from modern_yolonas import yolo_nas_s
from modern_yolonas.data.collate import detection_collate_fn
from modern_yolonas.data.transforms import (
    Compose,
    HorizontalFlip,
    HSVAugment,
    LetterboxResize,
    Normalize,
    RandomAffine,
)
from modern_yolonas.data.yolo import YOLODetectionDataset
from modern_yolonas.training.trainer import Trainer
from torch.utils.data import DataLoader

DATASET_DIR = Path("datasets/coco_animals")
OUTPUT_DIR = Path("runs/yolonas_s_voc")
INPUT_SIZE = 256
NUM_CLASSES = 5


@click.command()
@click.option("--epochs", "-e", default=30, type=int, help="Training epochs")
@click.option("--batch-size", "-b", default=8, type=int, help="Batch size")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
def main(epochs: int, batch_size: int, lr: float) -> None:
    if not DATASET_DIR.exists():
        logger.error(f"Dataset not found at {DATASET_DIR}. Run download_dataset.py first.")
        return

    logger.info(f"Fine-tuning YOLO-NAS-S on VOC-2007 ({epochs} epochs, bs={batch_size})")

    # Augmentations
    train_transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=INPUT_SIZE),
        Normalize(),
    ])

    val_transforms = Compose([
        LetterboxResize(target_size=INPUT_SIZE),
        Normalize(),
    ])

    # Datasets
    train_dataset = YOLODetectionDataset(
        root=str(DATASET_DIR),
        split="train",
        transforms=train_transforms,
        input_size=INPUT_SIZE,
    )

    val_dataset = YOLODetectionDataset(
        root=str(DATASET_DIR),
        split="val",
        transforms=val_transforms,
        input_size=INPUT_SIZE,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    # Load COCO pretrained backbone/neck, then swap heads for NUM_CLASSES
    coco_model = yolo_nas_s(pretrained=True, num_classes=80)
    model = yolo_nas_s(pretrained=False, num_classes=NUM_CLASSES)
    # Copy backbone + neck weights (skip head mismatches)
    pretrained_dict = {
        k: v for k, v in coco_model.state_dict().items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }
    model.load_state_dict(pretrained_dict, strict=False)
    del coco_model
    logger.info(f"Loaded {len(pretrained_dict)}/{len(model.state_dict())} pretrained params (heads re-initialized)")

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Trainer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        epochs=epochs,
        lr=lr,
        optimizer_name="adamw",
        weight_decay=1e-5,
        warmup_steps=500,
        use_amp=True,
        use_ema=True,
        output_dir=str(OUTPUT_DIR),
        device="cuda",
    )

    t0 = time.time()
    trainer.train()
    training_time = time.time() - t0

    # Collect GPU metrics
    peak_gpu_mb = 0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Save training summary
    summary = {
        "model": "YOLO-NAS-S",
        "input_size": INPUT_SIZE,
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
