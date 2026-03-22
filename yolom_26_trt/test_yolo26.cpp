#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "yolo26_src/yolo26_main.h"

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



int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_path>" << std::endl;
        return -1;
    }

    std::string engine_file = argv[1];
    std::string image_path = argv[2];

    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully: " << image.cols << "x" << image.rows << std::endl;

    // Initialize YOLO26 detector
    YOLO26 detector(640, 640, 1, 3, 1800, engine_file);  // 3*640*640 input size, 1800 output size (1*300*6), batch size 1

    std::cout << "Engine loaded successfully" << std::endl;

    // Perform detection
    std::vector<std::vector<float>> detections;
    if (!detector.infer_(image, detections)) {
        std::cerr << "Detection failed!" << std::endl;
        return -1;
    }

    std::cout << "Detection completed. Found " << detections.size() << " objects." << std::endl;

    // Draw detections on the image
    cv::Mat result_img = image.clone();

    for (size_t i = 0; i < detections.size(); ++i)
    {
        auto &single_res = detections[i];

        int name_idx = static_cast<int>(single_res[0]);
        float conf = single_res[1];
        float x1 = single_res[2];
        float y1 = single_res[3];
        float x2 = single_res[4];
        float y2 = single_res[5];

        // draw box
        cv::rectangle(
            result_img,
            cv::Rect(x1, y1, x2 - x1, y2 - y1),
            cv::Scalar(0, 255, 0),
            2
        );

        // label
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string label = std::string(COCO_NAME[name_idx]) + 
                            "(" + oss.str() + ")";

        cv::putText(
            result_img,
            label,
            cv::Point(x1, y1 - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1
        );
    }

    // Save the result
    std::string output_path = "result_" + std::to_string(cv::getTickCount()) + ".jpg";
    cv::imwrite(output_path, result_img);
    std::cout << "Result saved to: " << output_path << std::endl;

    return 0;
}