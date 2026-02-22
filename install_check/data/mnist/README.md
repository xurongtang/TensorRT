# Setting Up MNIST Samples

## Models

**mnist.onnx**: Opset 8, Retrieved from [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)

### Attribution
This model is originally from the ONNX Model Zoo and is licensed under the Apache License 2.0.
- **Original Source**: [ONNX Model Zoo - MNIST](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) ONNX contributors

## Run ONNX model with trtexec

* FP32 precisons with fixed batch size 1
  * `./trtexec --explicitBatch --onnx=mnist.onnx --workspace=1024`
* Other precisions
  * Add `--fp16` for FP16 and `--int8` for INT8.

## Run safety ONNX model with sampleSafeMNIST

* Build safe engine
  * `./sample_mnist_safe_build`
* Inference
  * `./sample_mnist_safe_infer`
* See sample READEME for more details.

## License

This model and associated files are distributed under the Apache License 2.0. See the `LICENSE` file for the full license text and the `NOTICE` file for attribution information.
