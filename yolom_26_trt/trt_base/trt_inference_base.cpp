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
    // In TensorRT 10.x, manual destroy() calls are deprecated
    // Resources are automatically managed by the library
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
}

bool TrtInferenceBase::loadEngine(const std::string& file)
{
    engineFile = file;
    
    // Read engine file
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

    // Create runtime
    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        delete[] trtModelStream;
        return false;
    }

    // Deserialize engine
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    delete[] trtModelStream;
    
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "Successfully loaded engine: " << file << std::endl;
    return true;
}

bool TrtInferenceBase::executeInference(void* inputBuffer, void* outputBuffer, int batchSize)
{
    if (!context || !engine) {
        std::cerr << "Execution context or engine is not initialized" << std::endl;
        return false;
    }

    // Get number of bindings to iterate through them
    int num_bindings = engine->getNbIOTensors();
    
    // Find input and output tensor names
    const char* inputName = nullptr;
    const char* outputName = nullptr;
    
    for (int i = 0; i < num_bindings; i++) {
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(engine->getIOTensorName(i));
        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            if (inputName == nullptr) {  // Take the first input
                inputName = engine->getIOTensorName(i);
            }
        } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (outputName == nullptr) {  // Take the first output
                outputName = engine->getIOTensorName(i);
            }
        }
    }
    
    if (!inputName || !outputName) {
        std::cerr << "Could not find input or output tensor names" << std::endl;
        return false;
    }
    
    context->setTensorAddress(inputName, inputBuffer);
    context->setTensorAddress(outputName, outputBuffer);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Execute inference
    bool status = context->enqueueV3(stream);

    if (!status) {
        std::cerr << "Failed to enqueue inference" << std::endl;
    } else {
        // Synchronize
        cudaStreamSynchronize(stream);
    }

    // Destroy stream
    cudaStreamDestroy(stream);

    return status;
}