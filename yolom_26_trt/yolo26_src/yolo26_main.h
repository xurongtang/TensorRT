
#ifndef YOLOM_H
#define YOLOM_H

#include "trt_base/trt_inference_base.h"
#include <vector>
#include <opencv2/opencv.hpp>

class YOLO26 : public TrtInferenceBase
{
    public:
        YOLO26(int input_height_ , int input_width_, int batchSize, int channels, int output_size, const std::string& engineFile);
        ~YOLO26();

        // Implement base class virtual functions
        bool preprocess() override;
        bool postprocess() override;
        bool infer() override;

        // Specific Method
        bool setInputData(const cv::Mat& input_image);
        std::vector<float> getOutputData() const { return outputData; }
        
        // 外部使用接口
        bool infer_(const cv::Mat& input_image, std::vector<std::vector<float>>& output_detect_result);

    private:
        // Input/output buffer on both CPU and GPU
        float* d_input{nullptr};
        float* h_output{nullptr};
        float* d_output{nullptr};
        
        // Input/output data
        std::vector<float> inputData;
        std::vector<float> outputData;
        
        // Dimensions
        int input_height;  // 640
        int input_width;   // 640
        int channels;      // 3
        int batchSize;        // 1

        // 
        int inputSize;
        int outputSize;

        // 模型输入输出的宽高比
        int orig_w_;
        int orig_h_;
        int pad_x_;
        int pad_y_;
        int scale_;

};

#endif // MLP_INFERENCE_H