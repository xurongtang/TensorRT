#include "trt_inference_base.h"
#include <iostream>
#include <fstream>
#include <cassert>

TrtInferenceBase::TrtInferenceBase() : 
    runtime(nullptr), 
    engine(nullptr), 
    context(nullptr), 
    initialized(false)
{
}

TrtInferenceBase::~TrtInferenceBase()
{
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
}

bool TrtInferenceBase::loadEngine(const std::string& file)
{
    engineFile = file;
    
    std::ifstream engineFileStream(file, std::ios::binary);
    if (!engineFileStream.good()) {
        std::cerr << "Error opening engine file: " << file << std::endl;
        return false;
    }

    engineFileStream.seekg(0, engineFileStream.end);
    size_t size = engineFileStream.tellg();
    engineFileStream.seekg(0, engineFileStream.beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    engineFileStream.read(trtModelStream, size);
    engineFileStream.close();

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        delete[] trtModelStream;
        return false;
    }

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    delete[] trtModelStream;
    
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "Successfully loaded engine: " << file << std::endl;
    return true;
}

std::vector<std::string> TrtInferenceBase::getInputTensorNames() const
{
    std::vector<std::string> names;
    if (!engine) return names;
    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            names.push_back(name);
        }
    }
    return names;
}

std::vector<std::string> TrtInferenceBase::getOutputTensorNames() const
{
    std::vector<std::string> names;
    if (!engine) return names;
    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            names.push_back(name);
        }
    }
    return names;
}

bool TrtInferenceBase::executeInference(void* inputBuffer, void* outputBuffer, int batchSize)
{
    if (!context || !engine) {
        std::cerr << "Execution context or engine is not initialized" << std::endl;
        return false;
    }

    const char* inputName = nullptr;
    const char* outputName = nullptr;
    
    int num_bindings = engine->getNbIOTensors();
    for (int i = 0; i < num_bindings; i++) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        if (ioMode == nvinfer1::TensorIOMode::kINPUT && !inputName) {
            inputName = name;
        } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT && !outputName) {
            outputName = name;
        }
    }
    
    if (!inputName || !outputName) {
        std::cerr << "Could not find input or output tensor names" << std::endl;
        return false;
    }
    
    context->setTensorAddress(inputName, inputBuffer);
    context->setTensorAddress(outputName, outputBuffer);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to enqueue inference" << std::endl;
    } else {
        cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
    return status;
}

bool TrtInferenceBase::executeInference(const std::unordered_map<std::string, void*>& bindings)
{
    if (!context || !engine) {
        std::cerr << "Execution context or engine is not initialized" << std::endl;
        return false;
    }

    // Set all tensor addresses from the bindings map
    for (const auto& kv : bindings) {
        context->setTensorAddress(kv.first.c_str(), kv.second);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Failed to enqueue inference" << std::endl;
    } else {
        cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
    return status;
}