#include "mlp_test/mlp_inference.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{ 
    // Create MLP inference instance with input size 10, output size 2, batch size 1
    MlpInference mlpInference(10, 2, 1);

    // Load the engine file
    if (!mlpInference.loadEngine("/home/rton/CppProj/TensorProj/tensorrt_inference_base/mlp_plus.engine")) // Using the same engine name as in mlp_plus.cpp
    {
        std::cout << "Failed to load engine" << std::endl;
        return -1;
    }

    // Define input data - same as in mlp_plus.cpp
    std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> output_data;

    std::cout << "Performing inference..." << std::endl;
    
    // Perform inference
    if (!mlpInference.infer(input_data, output_data))
    {
        std::cout << "Inference failed" << std::endl;
        return -1;
    }

    // Print results
    std::cout << "Input:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "[" << i << "]: " << input_data[i] << std::endl;
    }
    std::cout << "Input (cont.):" << std::endl;
    for (int i = 5; i < 10; i++) {
        std::cout << "[" << i << "]: " << input_data[i] << std::endl;
    }

    std::cout << "\nOutput:" << std::endl;
    for (size_t i = 0; i < output_data.size(); i++) {
        std::cout << "[" << i << "]: " << output_data[i] << std::endl;
    }

    std::cout << "\nInference completed successfully!" << std::endl;
    return 0;
}