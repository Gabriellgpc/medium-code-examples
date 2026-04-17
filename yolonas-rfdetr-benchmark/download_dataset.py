"""Download COCO-2017 animal subset and export to YOLO format for fine-tuning."""

import os
from pathlib import Path

import click
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four
import yaml
from loguru import logger

DATASET_DIR = Path("datasets/coco_animals")

# 5-class animal detection subset (a "custom domain" from COCO)
CLASSES = ["bird", "cat", "dog", "horse", "sheep"]


@click.command()
@click.option("--train-samples", default=200, type=int, help="Number of training images")
@click.option("--val-samples", default=50, type=int, help="Number of validation images")
def main(train_samples: int, val_samples: int) -> None:
    total = train_samples + val_samples
    logger.info(f"Preparing COCO animals dataset: {train_samples} train + {val_samples} val")
    logger.info(f"  Classes: {CLASSES}")

    # Load COCO-2017 val (already cached), filtered to animal classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        classes=CLASSES,
        max_samples=total,
        dataset_name=f"coco-animals-finetune-{total}",
    )

    # Random split into train/val
    train_ratio = train_samples / total
    four.random_split(dataset, {"train": train_ratio, "val": 1 - train_ratio})

    train_view = dataset.match_tags("train")
    val_view = dataset.match_tags("val")

    logger.info(f"  Train samples: {len(train_view)}")
    logger.info(f"  Val samples: {len(val_view)}")

    # Export to FiftyOne YOLO format (images/train, images/val, labels/train, labels/val)
    logger.info("Exporting train split...")
    train_view.export(
        export_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        split="train",
        classes=CLASSES,
    )

    logger.info("Exporting val split...")
    val_view.export(
        export_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        classes=CLASSES,
    )

    # Create RF-DETR compatible directory structure via symlinks
    # RF-DETR expects: train/images/, train/labels/, valid/images/, valid/labels/
    for split_src, split_dst in [("train", "train"), ("val", "valid")]:
        split_dir = DATASET_DIR / split_dst
        split_dir.mkdir(exist_ok=True)
        for sub in ["images", "labels"]:
            link = split_dir / sub
            target = os.path.relpath(DATASET_DIR / sub / split_src, split_dir)
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(target, str(link))

    # Create data.yaml for RF-DETR
    data_yaml = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(DATASET_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"Dataset exported to {DATASET_DIR}")
    logger.info(f"  Classes: {len(CLASSES)} ({', '.join(CLASSES)})")
    logger.info("  Compatible with both RF-DETR (train/valid dirs) and YOLO-NAS (images/labels dirs)")


if __name__ == "__main__":
    main()
