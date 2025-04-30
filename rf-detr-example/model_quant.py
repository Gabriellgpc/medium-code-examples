from pathlib import Path

import click
import nncf
import openvino as ov
import torch
from loguru import logger
from torchvision import datasets, transforms


class ImagesOnlyFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_path = [f for f in Path(root).rglob("*") if f.is_file()]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True),
    default=Path(__file__).parent / "output/ov_model/inference_model.xml",
    show_default=True,
    help="Path to the model to be quantized.",
)
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    default=Path(__file__).parent / "coco-2017-images",
    show_default=True,
    help="Path to the dataset for calibration.",
)
@click.option(
    "--output_path",
    type=click.Path(),
    default=Path(__file__).parent / "quant-output/quantized_model.xml",
    show_default=True,
    help="Path to save the quantized model.",
)
def nncf_ov_quantize(model_path: str, dataset_path: str, output_path: str):
    """
    Function to quantize the model using NNCF.
    :param model: The model to be quantized.
    :param dataset: The dataset for calibration.
    :return: The quantized model.
    """
    # Instantiate your uncompressed model
    logger.info(f"Loading model from {model_path}")
    model = ov.Core().read_model(model_path)

    # Provide validation part of the dataset to collect statistics
    # needed for the compression algorithm

    # Step 1: Create a dataset
    logger.info(f"Loading dataset from {dataset_path}")
    prep_transform = [
        transforms.ToTensor(),
        transforms.Resize((280, 280)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    val_dataset = ImagesOnlyFolder(
        dataset_path,
        transform=transforms.Compose(prep_transform)
    )
    logger.info(f"Loaded {len(val_dataset)} images from {dataset_path}")

    dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1)

    # Step 2: Initialize NNCF Dataset
    calibration_dataset = nncf.Dataset(dataset_loader)
    # Step 3: Run the quantization pipeline
    quantized_model = nncf.quantize(model, calibration_dataset)
    # Step 4: Save the quantized model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving quantized model to {output_path}")
    ov.save_model(quantized_model, str(output_path))


if __name__ == "__main__":
    nncf_ov_quantize()
