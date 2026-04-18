#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/yolom26_int8_batch.h"

std::vector<std::string> COCO_NAME = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

static void draw_detections(cv::Mat& img, const ImageDetections& dets)
{
    for (const auto& d : dets) {
        int   cid  = static_cast<int>(d[0]);
        float conf = d[1];
        float x1 = d[2], y1 = d[3], x2 = d[4], y2 = d[5];

        cv::rectangle(img, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 0), 2);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string label = COCO_NAME[cid] + std::string("(") + oss.str() + ")";
        cv::putText(img, label, cv::Point(x1, y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image1> [image2] ... [imageN]" << std::endl;
        std::cerr << "  Batch mode: provide multiple images to test batch inference" << std::endl;
        return -1;
    }

    std::string engine_file = argv[1];
    int batch_size = std::max(1, argc - 2);  // number of images provided

    // We initialize with a fixed batch size (e.g., 4) to demonstrate batch capability
    // Even if fewer images are provided, the remaining slots are zero-padded
    int max_batch = 4;
    if (batch_size > max_batch) {
        std::cerr << "Warning: max batch is " << max_batch << ", using that instead of " << batch_size << std::endl;
        batch_size = max_batch;
    }

    std::cout << "=== YOLO26 INT8 Batch Inference Test ===" << std::endl;
    std::cout << "Engine: " << engine_file << std::endl;
    std::cout << "Batch size: " << max_batch << std::endl;

    // Initialize detector with batch size
    YOLOM26Int8Batch detector(640, 640, max_batch, 3, engine_file);

    // Load images
    std::vector<cv::Mat> images;
    for (int i = 2; i < argc && i < 2 + max_batch; ++i) {
        cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << argv[i] << std::endl;
            return -1;
        }
        images.push_back(img);
        std::cout << "Loaded image " << (i - 1) << ": " << argv[i]
                  << " (" << img.cols << "x" << img.rows << ")" << std::endl;
    }

    if (images.empty()) {
        std::cerr << "No valid images provided!" << std::endl;
        return -1;
    }

    // ---- Single image test ----
    std::cout << "\n--- Single Image Inference ---" << std::endl;
    ImageDetections single_result;
    auto t1 = std::chrono::high_resolution_clock::now();
    bool ok = detector.infer_single(images[0], single_result, 0.25f);
    auto t2 = std::chrono::high_resolution_clock::now();
    float ms_single = std::chrono::duration<float, std::milli>(t2 - t1).count();

    if (!ok) {
        std::cerr << "Single inference failed!" << std::endl;
        return -1;
    }
    std::cout << "Single inference: " << ms_single << " ms, "
              << single_result.size() << " detections" << std::endl;

    // Draw & save single result
    cv::Mat single_img = images[0].clone();
    draw_detections(single_img, single_result);
    cv::imwrite("result_int8_single.jpg", single_img);
    std::cout << "Saved: result_int8_single.jpg" << std::endl;

    // ---- Batch inference test ----
    if (images.size() > 1) {
        std::cout << "\n--- Batch Inference (" << images.size() << " images) ---" << std::endl;
        std::vector<ImageDetections> batch_results;
        auto t3 = std::chrono::high_resolution_clock::now();
        ok = detector.infer_batch(images, batch_results, 0.25f);
        auto t4 = std::chrono::high_resolution_clock::now();
        float ms_batch = std::chrono::duration<float, std::milli>(t4 - t3).count();

        if (!ok) {
            std::cerr << "Batch inference failed!" << std::endl;
            return -1;
        }
        std::cout << "Batch inference: " << ms_batch << " ms total, "
                  << (ms_batch / images.size()) << " ms per image" << std::endl;

        for (size_t i = 0; i < batch_results.size(); ++i) {
            cv::Mat img = images[i].clone();
            draw_detections(img, batch_results[i]);
            std::string fname = "result_int8_batch_" + std::to_string(i) + ".jpg";
            cv::imwrite(fname, img);
            std::cout << "Image " << i << ": " << batch_results[i].size()
                      << " detections -> " << fname << std::endl;
        }
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}