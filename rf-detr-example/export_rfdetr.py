"""Export RFDETR models to ONNX and OpenVINO formats."""

from pathlib import Path

from loguru import logger


def save_labelmap_file(labelmap: list[str], output_dir: str) -> None:
    """Save the labelmap to a file."""
    labelmap_path = Path(output_dir) / "labelmap.txt"
    with labelmap_path.open("w") as file:
        file.write("\n".join(labelmap))
    logger.success(f"Labelmap file saved to {labelmap_path}")

def export_model(output_dir: Path) -> None:
    """Export the model to ONNX format."""
    import openvino as ov
    from rfdetr import RFDETRBase

    model = RFDETRBase(resolution=280)

    # Export the model to ONNX format
    output_dir.mkdir(exist_ok=True, parents=True)

    model.export(output_dir=output_dir.as_posix())

    onnx_path = output_dir / "inference_model.onnx"

    # Export to OpenVINO format
    ov_path = output_dir / "ov_model " / "inference_model.xml"
    if not ov_path.exists():
        logger.info("Exporting model to OpenVINO format")
        ov_model_dir = output_dir / "ov_model"
        ov_model_dir.mkdir(exist_ok=True, parents=True)

        ov_model = ov.convert_model(onnx_path, verbose=True)
        logger.info("Applying pruning transformation")
        ov.save_model(ov_model, Path(ov_model_dir) / "inference_model.xml")

    else:
        logger.warning(f"File {ov_path} already exists. Skipping export.")


if __name__ == "__main__":
    output_dir = Path("./output")
    export_model(output_dir)
