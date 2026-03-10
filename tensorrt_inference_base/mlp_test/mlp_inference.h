#ifndef MLP_INFERENCE_H
#define MLP_INFERENCE_H

#include "trt_base/trt_inference_base.h"
#include <vector>

class MlpInference : public TrtInferenceBase
{
private:
    // Input/output buffer on both CPU and GPU
    float* d_input{nullptr};
    float* h_output{nullptr};
    float* d_output{nullptr};
    
    // Input/output data
    std::vector<float> inputData;
    std::vector<float> outputData;
    
    // Dimensions
    int inputSize;
    int outputSize;
    int batchSize;

public:
    MlpInference(int inputSize = 10, int outputSize = 2, int batchSize = 1);
    ~MlpInference();

    // Implement base class virtual functions
    bool preprocess() override;
    bool postprocess() override;
    bool infer() override;

    // Specific methods for MLP
    bool setInputData(const std::vector<float>& data);
    std::vector<float> getOutputData() const { return outputData; }
    
    // Do inference with input data and return output
    bool infer(const std::vector<float>& input, std::vector<float>& output);
};

#endif // MLP_INFERENCE_H