# RF-DETR Exporting and Inference with OpenVINO

# Run benchmark

```bash
uv run benchmark_app -m <xml-model-path> -d GPU
```

# TODO
- [x] Code to export RF-DETR to ONNX & OpenVINO
- [x] Code to run inference for OpenVINO format
- [x] Code to quantize to int8 using NNCF

# References

- [Neural Network Compression Framework for enhanced OpenVINO™ inference](https://github.com/openvinotoolkit/nncf)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [RF-DETR](https://github.com/roboflow/rf-detr)
- [NNCF:post_training_quantization](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control/main.py)