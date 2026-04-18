#include "yolom26_int8_batch.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

// ============================================================
// Construction / Destruction
// ============================================================

YOLOM26Int8Batch::YOLOM26Int8Batch(int input_h, int input_w, int batch_size,
                                   int chans, const std::string& engineFile)
    : input_height(input_h)
    , input_width(input_w)
    , batchSize(batch_size)
    , channels(chans)
{
    singleInputSize  = channels * input_height * input_width;
    singleOutputSize = 300 * 6;  // YOLO26 one-to-one head: (num_dets=300, 6)
    totalInputSize   = batchSize * singleInputSize;
    totalOutputSize  = batchSize * singleOutputSize;

    // Allocate host buffers
    h_input  = new float[totalInputSize]();
    h_output = new float[totalOutputSize]();

    // Allocate device buffers
    cudaMalloc(reinterpret_cast<void**>(&d_input),  totalInputSize  * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), totalOutputSize * sizeof(float));

    // Load engine (must happen after buffer allocation)
    if (!loadEngine(engineFile)) {
        std::cerr << "Failed to load engine: " << engineFile << std::endl;
        return;
    }

    // Discover tensor names from engine
    auto inNames  = getInputTensorNames();
    auto outNames = getOutputTensorNames();
    if (!inNames.empty())  inputTensorName  = inNames[0];
    if (!outNames.empty()) outputTensorName = outNames[0];

    std::cout << "YOLOM26Int8Batch initialized. batch=" << batchSize
              << " input=" << inputTensorName
              << " output=" << outputTensorName << std::endl;
}

YOLOM26Int8Batch::~YOLOM26Int8Batch()
{
    if (d_input)    { cudaFree(d_input);    d_input = nullptr; }
    if (d_output)   { cudaFree(d_output);   d_output = nullptr; }
    if (h_input)    { delete[] h_input;  h_input = nullptr; }
    if (h_output)   { delete[] h_output; h_output = nullptr; }
}

// ============================================================
// Letterbox helper
// ============================================================

static cv::Mat letterbox(const cv::Mat& image, const cv::Size& new_shape,
                         const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    float width_ratio  = new_shape.width  / static_cast<float>(image.cols);
    float height_ratio = new_shape.height / static_cast<float>(image.rows);
    float scale = std::min(width_ratio, height_ratio);

    int new_w = static_cast<int>(image.cols * scale);
    int new_h = static_cast<int>(image.rows * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat canvas = cv::Mat::zeros(new_shape.height, new_shape.width, image.type());
    canvas.setTo(color);
    int top  = (new_shape.height - new_h) / 2;
    int left = (new_shape.width  - new_w) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    return canvas;
}

// ============================================================
// Preprocess a single image → CHW float buffer
// ============================================================

LetterboxInfo YOLOM26Int8Batch::preprocessImage(const cv::Mat& image, float* dst)
{
    LetterboxInfo info;
    info.orig_w = image.cols;
    info.orig_h = image.rows;

    float w_ratio = input_width  / static_cast<float>(info.orig_w);
    float h_ratio = input_height / static_cast<float>(info.orig_h);
    info.scale = std::min(w_ratio, h_ratio);

    int new_w = static_cast<int>(info.orig_w * info.scale);
    int new_h = static_cast<int>(info.orig_h * info.scale);
    info.pad_x = (input_width  - new_w) / 2;
    info.pad_y = (input_height - new_h) / 2;

    cv::Mat processed = letterbox(image, cv::Size(input_width, input_height));
    processed.convertTo(processed, CV_32F, 1.0f / 255.0f);

    // HWC → CHW
    std::vector<cv::Mat> ch_planes;
    cv::split(processed, ch_planes);
    int plane_size = input_width * input_height;
    for (int c = 0; c < channels; ++c) {
        memcpy(dst + c * plane_size, ch_planes[c].data, plane_size * sizeof(float));
    }
    return info;
}

// ============================================================
// Compute IoU of two boxes
// ============================================================
static float boxIou(float x1a, float y1a, float x2a, float y2a,
                    float x1b, float y1b, float x2b, float y2b)
{
    float ix1 = std::max(x1a, x1b);
    float iy1 = std::max(y1a, y1b);
    float ix2 = std::min(x2a, x2b);
    float iy2 = std::min(y2a, y2b);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    if (inter <= 0.f) return 0.f;
    float areaA = (x2a - x1a) * (y2a - y1a);
    float areaB = (x2b - x1b) * (y2b - y1b);
    return inter / (areaA + areaB - inter);
}

// ============================================================
// NMS (Non-Maximum Suppression)
// ============================================================
static void nms(ImageDetections& dets, float iou_threshold = 0.45f)
{
    if (dets.size() <= 1) return;

    // Sort by confidence (descending)
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a[1] > b[1];
    });

    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            // Only suppress same class
            if (static_cast<int>(dets[i][0]) != static_cast<int>(dets[j][0])) continue;
            float iou = boxIou(dets[i][2], dets[i][3], dets[i][4], dets[i][5],
                               dets[j][2], dets[j][3], dets[j][4], dets[j][5]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed detections
    ImageDetections kept;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!suppressed[i]) kept.push_back(std::move(dets[i]));
    }
    dets = std::move(kept);
}

// ============================================================
// Postprocess a single image's detections (with NMS)
// ============================================================

void YOLOM26Int8Batch::postprocessImage(const float* output, int num_detections,
                                         const LetterboxInfo& info,
                                         ImageDetections& result, float conf_threshold)
{
    result.clear();
    for (int i = 0; i < num_detections; ++i) {
        int base = i * 6;
        float x1   = output[base + 0];
        float y1   = output[base + 1];
        float x2   = output[base + 2];
        float y2   = output[base + 3];
        float conf = output[base + 4];
        int   cid  = static_cast<int>(output[base + 5]);

        if (conf < conf_threshold) continue;

        // Remove letterbox padding & rescale
        x1 = (x1 - info.pad_x) / info.scale;
        y1 = (y1 - info.pad_y) / info.scale;
        x2 = (x2 - info.pad_x) / info.scale;
        y2 = (y2 - info.pad_y) / info.scale;

        // Clip to original image bounds
        x1 = std::max(0.f, std::min(x1, static_cast<float>(info.orig_w)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(info.orig_h)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(info.orig_w)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(info.orig_h)));

        // Skip degenerate boxes
        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;

        result.push_back({static_cast<float>(cid), conf, x1, y1, x2, y2});
    }

    // Apply NMS to remove overlapping detections
    nms(result, 0.45f);
}

// ============================================================
// Batch inference
// ============================================================

bool YOLOM26Int8Batch::infer_batch(const std::vector<cv::Mat>& images,
                                    std::vector<ImageDetections>& results,
                                    float conf_threshold)
{
    if (images.empty()) {
        std::cerr << "No images provided" << std::endl;
        return false;
    }
    if (static_cast<int>(images.size()) > batchSize) {
        std::cerr << "Number of images (" << images.size()
                  << ") exceeds batch size (" << batchSize << ")" << std::endl;
        return false;
    }

    results.resize(images.size());

    /*
    ## 真正的多 batch 并行推理
    ONNX 模型已导出为 batch=4 的静态尺寸：输入 `[4, 3, 640, 640]`，输出 `[4, 300, 6]`。
    `model.23` 内部的 `TopK` 操作使用 `axis=-1`，在每个图像的检测维度上独立操作，
    因此每张图各自产生 300 个检测结果，不会跨图共享检测名额。

    推理流程：
    1. 预处理所有图像 → 填入 batch 输入缓冲区
    2. 一次性 H2D 传输整个 batch
    3. 执行一次推理（所有图像并行处理）
    4. 一次性 D2H 传输整个 batch 输出
    5. 对每张图的输出切片分别做后处理
    */

    int validBatch = static_cast<int>(images.size());

    // 1. 预处理所有图像，填入 batch 缓冲区
    std::vector<LetterboxInfo> infos(validBatch);
    memset(h_input, 0, totalInputSize * sizeof(float));  // 清零整个缓冲区

    for (int i = 0; i < validBatch; ++i) {
        if (images[i].empty()) {
            std::cerr << "Image " << i << " is empty!" << std::endl;
            return false;
        }
        infos[i] = preprocessImage(images[i], h_input + i * singleInputSize);
    }

    // 2. H2D - 传输整个 batch
    cudaMemcpy(d_input, h_input, totalInputSize * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 执行推理（所有图像并行）
    std::unordered_map<std::string, void*> bindings;
    bindings[inputTensorName]  = d_input;
    bindings[outputTensorName] = d_output;

    if (!executeInference(bindings)) {
        std::cerr << "Batch inference execution failed!" << std::endl;
        return false;
    }

    // 4. D2H - 读取整个 batch 输出
    cudaMemcpy(h_output, d_output, totalOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. 对每张图的输出切片做后处理
    for (int i = 0; i < validBatch; ++i) {
        postprocessImage(h_output + i * singleOutputSize, 300,
                         infos[i], results[i], conf_threshold);
    }

    return true;
}

// ============================================================
// Single image convenience wrapper
// ============================================================

bool YOLOM26Int8Batch::infer_single(const cv::Mat& image,
                                     ImageDetections& result,
                                     float conf_threshold)
{
    std::vector<cv::Mat> imgs = {image};
    std::vector<ImageDetections> results;
    if (!infer_batch(imgs, results, conf_threshold)) return false;
    result = std::move(results[0]);
    return true;
}

// ============================================================
// Base class virtual implementations (use batch internally)
// ============================================================

bool YOLOM26Int8Batch::preprocess()
{
    // Handled inside infer_batch / infer_single
    return true;
}

bool YOLOM26Int8Batch::postprocess()
{
    if (!currentResults_) return false;
    for (int i = 0; i < currentBatch_; ++i) {
        postprocessImage(h_output + i * singleOutputSize, 300,
                         currentLetterboxInfos_[i],
                         (*currentResults_)[i], currentConfThreshold_);
    }
    return true;
}

bool YOLOM26Int8Batch::infer()
{
    cudaMemcpy(d_input, h_input, totalInputSize * sizeof(float), cudaMemcpyHostToDevice);

    std::unordered_map<std::string, void*> bindings;
    bindings[inputTensorName]  = d_input;
    bindings[outputTensorName] = d_output;

    if (!executeInference(bindings)) return false;

    cudaMemcpy(h_output, d_output, totalOutputSize * sizeof(float), cudaMemcpyDeviceToHost);
    return postprocess();
}