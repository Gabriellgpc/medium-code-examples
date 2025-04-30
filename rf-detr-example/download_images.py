import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == "__main__":

    # Load the dataset from fiftyone
    # Download the COCO-2017 validation split and load it into FiftyOne
    dataset = foz.load_zoo_dataset("coco-2017",
                                   split="validation",
                                   max_samples=300)
    # (optional) make it persistent
    # dataset.persistent = True

    # (optional) Visualize it in the App
    # session = fo.launch_app(dataset)
    # session.wait()

    # Save images to disk
    dataset.export(
        export_dir="coco-2017-images",
        dataset_type=fo.types.ImageDirectory,
        progress=True,
    )

    # TODO From fiftyone to disk -> load using usual NNCF flow