#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolo26_main.h"
#include <algorithm>
#include <cmath>

YOLO26::YOLO26(int input_height_ , 
               int input_width_, 
               int batchSize, 
               int channels, 
               int output_size, 
               const std::string& engineFile) {

    this->input_height = input_height_;
    this->input_width = input_width_;
    this->batchSize = batchSize;
    this->channels = channels;
    // Calculate sizes based on model dimensions
    // Model output dimension is (1, 300, 6), so total outputs = 1 * 300 * 6 = 1800
    
    this->inputSize = input_width_ * input_height_ * channels;
    this->outputSize = 1 * 300 * 6; // Fixed to match model output
    
    // Allocate host memory for output
    h_output = new float[outputSize]();
    
    // Allocate device memory for input and output
    cudaMalloc(reinterpret_cast<void**>(&d_input), inputSize * batchSize * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), outputSize * sizeof(float));

    // load engine model
    if (!loadEngine(engineFile)) {
        std::cerr << "Failed to load engine file: " << engineFile << std::endl;
        return;
    }
}

YOLO26::~YOLO26()
{
    // Free allocated memory
    if(d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if(h_output) {
        delete[] h_output;
        h_output = nullptr;
    }
    if(d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

cv::Mat letterbox(const cv::Mat& image, const cv::Size& new_shape = cv::Size(640, 640), const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    cv::Mat resized_img;
    float width_ratio = new_shape.width / (float)image.cols;
    float height_ratio = new_shape.height / (float)image.rows;
    
    // Use the smaller ratio to maintain aspect ratio
    float scale = std::min(width_ratio, height_ratio);
    
    int new_width = (int)(image.cols * scale);
    int new_height = (int)(image.rows * scale);
    
    // Resize the image
    cv::resize(image, resized_img, cv::Size(new_width, new_height));
    
    // Create canvas with target size
    cv::Mat canvas = cv::Mat::zeros(new_shape.height, new_shape.width, image.type());
    canvas.setTo(color);
    
    // Calculate position to paste the resized image
    int top_padding = (new_shape.height - new_height) / 2;
    int left_padding = (new_shape.width - new_width) / 2;
    
    // Paste the resized image onto the canvas
    cv::Rect roi(left_padding, top_padding, new_width, new_height);
    resized_img.copyTo(canvas(roi));
    
    return canvas;
}

bool YOLO26::setInputData(const cv::Mat& input_image) 
{
    if (input_image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return false;
    }

    orig_w_ = input_image.cols;
    orig_h_ = input_image.rows;

    float width_ratio = input_width / (float)orig_w_;
    float height_ratio = input_height / (float)orig_h_;

    scale_ = std::min(width_ratio, height_ratio);

    int new_w = int(orig_w_ * scale_);
    int new_h = int(orig_h_ * scale_);

    pad_x_ = (input_width - new_w) / 2;
    pad_y_ = (input_height - new_h) / 2;

    cv::Mat processed_img = letterbox(input_image, cv::Size(input_width, input_height));
    processed_img.convertTo(processed_img, CV_32F, 1.0f / 255.0f);

    // HWC to CHW format
    std::vector<cv::Mat> channels;
    cv::split(processed_img, channels);
    
    int imgSize = input_width * input_height;
    inputData.resize(3 * imgSize);
    
    // Copy channel by channel (CHW format)
    int offset = 0;
    for (int c = 0; c < 3; ++c) {
        memcpy(&inputData[offset], channels[c].data, imgSize * sizeof(float));
        offset += imgSize;
    }

    // Copy input data to GPU
    cudaMemcpy(d_input, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice);

    return true;
}

bool YOLO26::preprocess() 
{
    // Preprocessing is done in setInputData method
    // This method can be extended for additional preprocessing steps if needed
    return true;
}

bool YOLO26::postprocess() 
{
    // Copy output from GPU to host
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Resize output vector to match the model output size
    outputData.resize(outputSize);
    memcpy(outputData.data(), h_output, outputSize * sizeof(float));

    return true;
}

bool YOLO26::infer()
{
    // Execute the inference using the base class method
    if (!executeInference(d_input, d_output, batchSize)) {
        std::cerr << "Inference failed!" << std::endl;
        return false;
    }

    // Post-process the output
    if (!postprocess()) {
        std::cerr << "Post-processing failed!" << std::endl;
        return false;
    }

    return true;
}

bool YOLO26::infer_(const cv::Mat& input_image, std::vector<std::vector<float>>& output_detect_result)
{
    // Set input data
    if (!setInputData(input_image)) {
        std::cerr << "Setting input data failed!" << std::endl;
        return false;
    }

    // Perform inference
    if (!infer()) {
        std::cerr << "Inference execution failed!" << std::endl;
        return false;
    }

    // Process detection results from output data
    // Output format is (batch, num_detections, 6) where 6 = [x1, y1, x2, y2, confidence, class_id]
    output_detect_result.clear();
    
    for (int i = 0; i < 300; ++i) { // Max 300 detections
        int base_idx = i * 6;
        
        // Extract bounding box coordinates, confidence, and class
        float x1 = outputData[base_idx];
        float y1 = outputData[base_idx + 1];
        float x2 = outputData[base_idx + 2];
        float y2 = outputData[base_idx + 3];
        float conf = outputData[base_idx + 4];
        int class_id = static_cast<int>(outputData[base_idx + 5]);
        
        // Apply confidence threshold (e.g., 0.5)
        if (conf > 0.5f)
        {
            // remove padding
            x1 -= pad_x_;
            y1 -= pad_y_;
            x2 -= pad_x_;
            y2 -= pad_y_;

            // divide by scale
            x1 /= scale_;
            y1 /= scale_;
            x2 /= scale_;
            y2 /= scale_;

            // clip to image
            x1 = std::max(0.f, std::min(x1, (float)orig_w_));
            y1 = std::max(0.f, std::min(y1, (float)orig_h_));
            x2 = std::max(0.f, std::min(x2, (float)orig_w_));
            y2 = std::max(0.f, std::min(y2, (float)orig_h_));

            std::vector<float> single_dect_res;
            single_dect_res.push_back(static_cast<float>(class_id));
            single_dect_res.push_back(conf);
            single_dect_res.push_back(x1);
            single_dect_res.push_back(y1);
            single_dect_res.push_back(x2);
            single_dect_res.push_back(y2);

            output_detect_result.push_back(single_dect_res);
        }
    }

    return true;
}