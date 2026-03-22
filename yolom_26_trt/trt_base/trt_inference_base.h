#ifndef TRT_INFERENCE_BASE_H
#define TRT_INFERENCE_BASE_H

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <string>
#include <iostream>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity reportableSeverity{ nvinfer1::ILogger::Severity::kWARNING };

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= reportableSeverity)
        {
            switch (severity)
            {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
            }
            std::cerr << msg << std::endl;
        }
    }
};

class TrtInferenceBase
{
protected:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    Logger logger;
    
    std::string engineFile;
    bool initialized;

    // Protected method to execute the actual inference
    bool executeInference(void* inputBuffer, void* outputBuffer, int batchSize);

public:
    TrtInferenceBase();
    virtual ~TrtInferenceBase();

    // Interface functions
    bool loadEngine(const std::string& engineFile);
    virtual bool infer() = 0;

    // Virtual functions for preprocessing and postprocessing
    virtual bool preprocess() = 0;
    virtual bool postprocess() = 0;

    // Utility functions
    bool isInitialized() const { return initialized; }
    nvinfer1::IExecutionContext* getContext() { return context; }
    nvinfer1::ICudaEngine* getEngine() { return engine; }
};

#endif // TRT_INFERENCE_BASE_H