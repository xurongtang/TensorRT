#include "tensorrt_infer.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cstring>  // for std::memcpy

namespace tensorrt {

// 辅助函数：计算 Dims 的 volume（避免依赖 samplesCommon::volume）
inline int64_t getVolume(const nvinfer1::Dims& dims) {
    int64_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1) continue;  // 跳过动态维度标记
        vol *= dims.d[i];
    }
    return vol;
}

TensorRTInfer::TensorRTInfer(const Params& params) 
    : mParams(params), mRuntime(nullptr), mEngine(nullptr), mContext(nullptr),
      mHostBufferInput(nullptr), mHostBufferOutput(nullptr) {}

TensorRTInfer::~TensorRTInfer() {
    // 释放设备内存
    for (auto& [name, ptr] : mBuffers) {
        if (ptr) cudaFree(ptr);
    }
    mBuffers.clear();
    
    // 释放主机 pinned 内存
    if (mHostBufferInput) cudaFreeHost(mHostBufferInput);
    if (mHostBufferOutput) cudaFreeHost(mHostBufferOutput);
    
    // RAII 会自动释放 mContext/mEngine/mRuntime
}

bool TensorRTInfer::buildEngine() {
    if (!loadModel()) return false;
    if (!parseOnnxModel()) return false;
    if (!buildCudaEngine()) return false;
    return true;
}

bool TensorRTInfer::infer() {
    return runInference();
}

bool TensorRTInfer::setInputData(const std::vector<float>& inputData) {
    if (inputData.empty()) return false;
    
    size_t expectedSize = 1;
    for (int i = 0; i < mInputDims.nbDims; ++i) {
        if (mInputDims.d[i] > 0) expectedSize *= mInputDims.d[i];
    }
    
    assert(inputData.size() == expectedSize && "Input size mismatch");

    // 拷贝到主机 pinned 内存（如果已分配）
    if (mHostBufferInput) {
        std::memcpy(mHostBufferInput, inputData.data(), inputData.size() * sizeof(float));
        // 拷贝到设备
        cudaMemcpy(mBuffers["input"], mHostBufferInput, inputData.size() * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        // 直接拷贝到设备
        cudaMemcpy(mBuffers["input"], inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    return true;
}

std::vector<float> TensorRTInfer::getOutputData() const {
    size_t outputSize = 1;
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
        if (mOutputDims.d[i] > 0) outputSize *= mOutputDims.d[i];
    }
    
    std::vector<float> outputData(outputSize);
    
    // 从设备拷贝到主机
    auto it = mBuffers.find("output");
    if (it != mBuffers.end() && it->second) {
        cudaMemcpy(outputData.data(), it->second, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return outputData;
}

bool TensorRTInfer::loadModel() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        sample::gLogError << "Failed to create IBuilder." << std::endl;
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    if (!network) {
        sample::gLogError << "Failed to create INetworkDefinition." << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        sample::gLogError << "Failed to create IBuilderConfig." << std::endl;
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        sample::gLogError << "Failed to create ONNX parser." << std::endl;
        return false;
    }

    auto parsed = parser->parseFromFile(mParams.onnxModelPath.c_str(), 
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        sample::gLogError << "Failed to parse ONNX model: " << mParams.onnxModelPath << std::endl;
        return false;
    }

    if (mParams.dlaCore != -1) {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    }

    auto serializedNetwork = builder->buildSerializedNetwork(*network, *config);
    if (!serializedNetwork) {
        sample::gLogError << "Failed to build serialized network." << std::endl;
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()),
        [](nvinfer1::IRuntime* ptr) { if (ptr) delete ptr; });
    if (!mRuntime) {
        sample::gLogError << "Failed to create IRuntime." << std::endl;
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(serializedNetwork->data(), serializedNetwork->size()),
        [](nvinfer1::ICudaEngine* ptr) { if (ptr) delete ptr; });
    if (!mEngine) {
        sample::gLogError << "Failed to deserialize CUDA engine." << std::endl;
        return false;
    }

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ptr) { if (ptr) delete ptr; });
    if (!mContext) {
        sample::gLogError << "Failed to create IExecutionContext." << std::endl;
        return false;
    }

    // ✅ 获取输入输出维度（新 API）
    if (mEngine->getNbIOTensors() < 2) {
        sample::gLogError << "Expected at least 2 IO tensors, got: " << mEngine->getNbIOTensors() << std::endl;
        return false;
    }
    
    const char* inputName = mEngine->getIOTensorName(0);
    const char* outputName = mEngine->getIOTensorName(1);
    
    if (!inputName || !outputName) {
        sample::gLogError << "Failed to get IO tensor names" << std::endl;
        return false;
    }
    
    mInputDims = mContext->getTensorShape(inputName);
    mOutputDims = mContext->getTensorShape(outputName);
    
    sample::gLogInfo << "Input tensor: " << inputName << ", dims: " << mInputDims << std::endl;
    sample::gLogInfo << "Output tensor: " << outputName << ", dims: " << mOutputDims << std::endl;

    return true;
}

bool TensorRTInfer::parseOnnxModel() {
    // 预留扩展点
    return true;
}

bool TensorRTInfer::buildCudaEngine() {
    // ✅ 遍历 IO tensors 并分配设备内存（新 API）
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i) {
        const char* tensorName = mEngine->getIOTensorName(i);  // 🔥 修复：正确声明变量
        if (!tensorName) continue;
        
        auto dims = mContext->getTensorShape(tensorName);
        int64_t volume = getVolume(dims);  // 使用本地辅助函数
        size_t bindingSize = volume * sizeof(float);  // 假设 FP32，如需支持其他类型请用 getTensorDataType

        void* deviceBuffer = nullptr;
        cudaError_t err = cudaMalloc(&deviceBuffer, bindingSize);
        if (err != cudaSuccess || !deviceBuffer) {
            sample::gLogError << "Failed to allocate device buffer for [" << tensorName 
                            << "]: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // ✅ 用 std::string 作为 key 避免 const char* 悬空问题
        mBuffers[std::string(tensorName)] = deviceBuffer;
        sample::gLogInfo << "Allocated [" << tensorName << "]: " << bindingSize << " bytes" << std::endl;
    }

    // 分配主机 pinned 内存（可选，用于加速 H2D/D2H 拷贝）
    size_t inputVol = getVolume(mInputDims);
    size_t outputVol = getVolume(mOutputDims);
    
    if (inputVol > 0) {
        cudaMallocHost(&mHostBufferInput, inputVol * sizeof(float));
    }
    if (outputVol > 0) {
        cudaMallocHost(&mHostBufferOutput, outputVol * sizeof(float));
    }

    return true;
}

bool TensorRTInfer::runInference() {
    if (!mContext || !mEngine) {
        sample::gLogError << "Context or engine not initialized" << std::endl;
        return false;
    }
    
    // ✅ 手动构建 bindings 数组 + 绑定 tensor 地址（新 API 核心）
    std::vector<void*> bindings;
    bindings.reserve(mEngine->getNbIOTensors());
    
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i) {
        const char* name = mEngine->getIOTensorName(i);
        if (!name) continue;
        
        std::string tensorName(name);
        auto it = mBuffers.find(tensorName);
        if (it == mBuffers.end()) {
            sample::gLogError << "Buffer not found for tensor: " << tensorName << std::endl;
            return false;
        }
        
        void* buffer = it->second;
        
        // 🔥 新 API 必须：显式绑定 tensor 地址
        mContext->setTensorAddress(name, buffer);
        bindings.push_back(buffer);
    }
    
    // ✅ 执行推理
    bool status = mContext->executeV2(bindings.data());
    if (!status) {
        sample::gLogError << "Inference execution failed!" << std::endl;
        return false;
    }
    
    // 同步确保结果就绪
    cudaStreamSynchronize(0);
    
    return true;  // 🔥 修复：正确的返回值位置
}

}  // namespace tensorrt