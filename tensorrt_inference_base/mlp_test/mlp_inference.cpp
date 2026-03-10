#include "mlp_inference.h"
#include <chrono>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cassert>

MlpInference::MlpInference(int inputSize_, int outputSize_, int batchSize_) :
    inputSize(inputSize_), outputSize(outputSize_), batchSize(batchSize_)
{
    // Allocate GPU and CPU buffers
    cudaMalloc(reinterpret_cast<void**>(&d_input), batchSize * inputSize * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), batchSize * outputSize * sizeof(float));
    h_output = new float[batchSize * outputSize];
}

MlpInference::~MlpInference()
{
    // Free allocated memory
    if (d_input) {
        cudaFree(d_input);
    }
    if (d_output) {
        cudaFree(d_output);
    }
    if (h_output) {
        delete[] h_output;
    }
}

bool MlpInference::preprocess()
{
    // Copy input data to GPU
    if (inputData.size() != static_cast<size_t>(batchSize * inputSize)) {
        std::cerr << "Input data size mismatch!" << std::endl;
        return false;
    }

    cudaMemcpy(d_input, inputData.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

bool MlpInference::postprocess()
{
    // Copy output from GPU to CPU
    cudaMemcpy(h_output, d_output, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Store output data
    outputData.resize(batchSize * outputSize);
    for (int i = 0; i < batchSize * outputSize; ++i) {
        outputData[i] = h_output[i];
    }

    return true;
}

bool MlpInference::infer()
{
    // Preprocess input
    if (!preprocess()) {
        std::cerr << "Preprocessing failed!" << std::endl;
        return false;
    }

    // Execute inference using the base class method
    bool status = executeInference(d_input, d_output, batchSize);

    if (!status) {
        std::cerr << "Inference execution failed!" << std::endl;
        return false;
    }

    // Postprocess output
    if (!postprocess()) {
        std::cerr << "Postprocessing failed!" << std::endl;
        return false;
    }

    return true;
}

bool MlpInference::setInputData(const std::vector<float>& data)
{
    if (data.size() != static_cast<size_t>(batchSize * inputSize)) {
        std::cerr << "Input data size mismatch! Expected: " << batchSize * inputSize 
                  << ", Got: " << data.size() << std::endl;
        return false;
    }

    inputData = data;
    return true;
}

bool MlpInference::infer(const std::vector<float>& input, std::vector<float>& output)
{
    if (!setInputData(input)) {
        return false;
    }

    if (!infer()) {
        return false;
    }

    output = getOutputData();
    return true;
}