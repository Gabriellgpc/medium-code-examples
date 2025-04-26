from pathlib import Path

import click
import nncf
import openvino as ov
import torch
from loguru import logger
from torchvision import datasets, transforms

# Dataset from fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# Dataset from HF
# Increase both connection and read timeout values (in seconds)
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # default is 10
# os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"      # metadata fetch timeout
#!pip install python-dotenv
# dataset = fouh.load_from_hub("dgural/bdd100k", persistent=True) #, overwrite=True)

@click.command()
@click.option('--model_path',
                type=click.Path(exists=True),
                default=Path(__file__).parent / "output/ov_model/inference_model.xml",
                show_default=True,
                help='Path to the model to be quantized.')
def nncf_ov_quantize(model_path: str):
    """
    Function to quantize the model using NNCF.
    :param model: The model to be quantized.
    :param dataset: The dataset for calibration.
    :return: The quantized model.
    """
    # Instantiate your uncompressed model
    logger.info(f"Loading model from {model_path}")
    model = ov.Core().read_model(model_path)

    # Provide validation part of the dataset to collect statistics needed for the compression algorithm
    # val_dataset = datasets.ImageFolder("/path",
    #                                    transform=transforms.Compose([transforms.ToTensor()]))
    # dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    dataset_loader = ...

    # Step 1: Initialize transformation function
    def transform_fn(data_item):
        images, _ = data_item
        return images

    # Step 2: Initialize NNCF Dataset
    calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
    # Step 3: Run the quantization pipeline
    quantized_model = nncf.quantize(model, calibration_dataset)
    # Step 4: Save the quantized model
    ov.save_model(quantized_model, "quantized_model.xml", model_name="quantized_model")

if __name__ == "__main__":

    # Load the dataset from fiftyone
    # Download the COCO-2017 validation split and load it into FiftyOne
    dataset = foz.load_zoo_dataset("coco-2017",
                                   split="validation",
                                   max_samples=300)
    # and make it persistent
    # dataset.persistent = True

    # Visualize it in the App
    session = fo.launch_app(dataset)
    session.wait()

    # TODO From fiftyone to disk -> load using usual NNCF flow