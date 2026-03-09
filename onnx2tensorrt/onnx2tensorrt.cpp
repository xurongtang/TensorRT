#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// Custom logger class for TensorRT
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // Only print messages with severity greater than warning
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cout << "[" << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "ms] ";
            
            switch (severity)
            {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: std::cout << "INTERNAL_ERROR: "; break;
                case nvinfer1::ILogger::Severity::kERROR: std::cout << "ERROR: "; break;
                case nvinfer1::ILogger::Severity::kWARNING: std::cout << "WARNING: "; break;
                case nvinfer1::ILogger::Severity::kINFO: std::cout << "INFO: "; break;
                default: break;
            }
            std::cout << msg << std::endl;
        }
    }
};

void convertOnnxToTensorRT(const std::string& onnxModelPath, const std::string& enginePath)
{
    // Create a TensorRT logger object
    Logger logger;

    // Create a TensorRT builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return;
    }

    // Create a network definition
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        return;
    }

    // Create a builder configuration
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cerr << "Failed to create builder config" << std::endl;
        return;
    }

    // Create an ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        std::cerr << "Failed to create parser" << std::endl;
        return;
    }

    // Parse the ONNX model
    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return;
    }

    // Enable FP16 if supported (optional for better performance)
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Build the engine
    std::cout << "Building engine..." << std::endl;
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cerr << "Failed to build engine" << std::endl;
        return;
    }

    // Write the serialized engine to file
    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Failed to open engine file for writing" << std::endl;
        return;
    }

    engineFile.write(static_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    std::cout << "Successfully converted ONNX model to TensorRT engine: " << enginePath << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <onnx_model_path> <engine_path>" << std::endl;
        return -1;
    }

    convertOnnxToTensorRT(argv[1], argv[2]);

    return 0;
}