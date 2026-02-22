#ifndef TENSORRT_INFER_H
#define TENSORRT_INFER_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace tensorrt {

class TensorRTInfer {
public:
    struct Params {
        std::string onnxModelPath;  // ONNX 模型文件路径
        std::vector<std::string> inputTensorNames;  // 输入张量名称
        std::vector<std::string> outputTensorNames;  // 输出张量名称
        std::vector<std::string> dataDirs;  // 数据目录路径
        int dlaCore = -1;  // 使用 DLA 核心（可选）
        std::string timingCacheFile;  // 定时缓存文件路径（可选）
    };

    TensorRTInfer(const Params& params);
    ~TensorRTInfer();

    // 构建网络和引擎
    bool buildEngine();

    // 进行推理
    bool infer();

    // 设置输入数据
    bool setInputData(const std::vector<float>& inputData);

    // 获取推理结果
    std::vector<float> getOutputData() const;

private:
    bool loadModel();
    bool parseOnnxModel();
    bool buildCudaEngine();
    bool runInference();

    Params mParams;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    std::unordered_map<std::string, void*> mBuffers;
    void* mHostBufferInput;
    void* mHostBufferOutput;
};

}  // namespace tensorrt

#endif  // TENSORRT_INFER_H