#ifndef YOLOM26_INT8_BATCH_H
#define YOLOM26_INT8_BATCH_H

#include "trt_base/trt_inference_base.h"
#include <vector>
#include <opencv2/opencv.hpp>

// Per-image detection result: [class_id, confidence, x1, y1, x2, y2]
typedef std::vector<float> Detection;
// Per-image detections
typedef std::vector<Detection> ImageDetections;

struct LetterboxInfo {
    int orig_w;
    int orig_h;
    int pad_x;
    int pad_y;
    float scale;
};

class YOLOM26Int8Batch : public TrtInferenceBase
{
public:
    YOLOM26Int8Batch(int input_h, int input_w, int batchSize, int channels,
                     const std::string& engineFile);
    ~YOLOM26Int8Batch();

    // Implement base class virtual functions
    bool preprocess() override;
    bool postprocess() override;
    bool infer() override;

    // Batch inference: process multiple images at once
    // images: vector of input images (size should be <= batchSize)
    // results: per-image detection results
    bool infer_batch(const std::vector<cv::Mat>& images,
                     std::vector<ImageDetections>& results,
                     float conf_threshold = 0.5f);

    // Single image inference (convenience wrapper, pads remaining batch slots with zeros)
    bool infer_single(const cv::Mat& image,
                      ImageDetections& result,
                      float conf_threshold = 0.5f);

private:
    // GPU buffers
    float* d_input{nullptr};
    float* d_output{nullptr};

    // CPU output buffer
    float* h_output{nullptr};

    // Host input staging buffer (for batch)
    float* h_input{nullptr};

    // Dimensions
    int input_height;   // 640
    int input_width;    // 640
    int channels;       // 3
    int batchSize;

    // Computed sizes
    int singleInputSize;   // channels * H * W
    int singleOutputSize;  // 300 * 6
    int totalInputSize;    // batchSize * singleInputSize
    int totalOutputSize;   // batchSize * singleOutputSize

    // Input/output tensor names (discovered from engine)
    std::string inputTensorName;
    std::string outputTensorName;

    // Preprocess a single image: letterbox + normalize + HWC->CHW
    // Returns the letterbox info for coordinate mapping
    LetterboxInfo preprocessImage(const cv::Mat& image, float* dst);

    // Postprocess a single image's output: filter by confidence, map coords back
    void postprocessImage(const float* output, int num_detections,
                          const LetterboxInfo& info,
                          ImageDetections& result, float conf_threshold);

    // Internal state for base class infer() — not used in batch mode
    std::vector<LetterboxInfo> currentLetterboxInfos_;
    int currentBatch_{0};
    float currentConfThreshold_{0.1f};
    std::vector<ImageDetections>* currentResults_{nullptr};
};

#endif // YOLOM26_INT8_BATCH_H