#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorrt_infer.h"  // 假设你已经有 TensorRTInfer 类

// 非极大值抑制（NMS）
std::vector<int> nms(std::vector<cv::Rect>& boxes, 
                     std::vector<float>& scores, 
                     float iouThreshold,
                     float scoreThreshold = 0.5f) {
    std::vector<int> indices;  // ✅ 必须是 std::vector<int>，不能是 cv::Mat
    // 边界检查
    if (boxes.empty() || scores.empty() || boxes.size() != scores.size()) {
        return indices;
    }
    // ✅ 正确调用：最后一个参数是 vector<int>&
    cv::dnn::NMSBoxes(
        boxes, 
        scores, 
        scoreThreshold,   // 置信度阈值
        iouThreshold,     // IOU 阈值
        indices           // 输出：保留的索引
    );
    // 直接返回 indices（RVO 优化，无额外拷贝）
    return indices;
}

// 后处理：YOLOv8
void postprocessYOLOv8(const std::vector<float>& output, const int inputWidth, 
                       const int inputHeight, const float confThreshold, 
                       const float iouThreshold, std::vector<cv::Rect>& boxes, 
                       std::vector<int>& classIds, std::vector<float>& scores) {
    int numClasses = 80;  // YOLOv8 的类别数，通常为 80，可以根据需要调整
    int gridSize = 13;    // 假设输出是 13x13 格式的 grid（可以根据模型调整）
    int stride = 32;      // 假设 YOLOv8 使用的 stride（可以根据模型调整）

    for (int i = 0; i < output.size() / (numClasses + 5); i++) {
        float* outputPtr = (float*)&output[i * (numClasses + 5)];

        float conf = outputPtr[4];
        if (conf < confThreshold) {
            continue;  // 过滤掉低置信度框
        }

        int classIdx = -1;
        float maxConf = 0.0f;
        for (int j = 5; j < 5 + numClasses; j++) {
            if (outputPtr[j] > maxConf) {
                maxConf = outputPtr[j];
                classIdx = j - 5;
            }
        }

        if (classIdx == -1) {
            continue;  // 没有找到有效的分类
        }

        // 解析边界框坐标
        float x = (outputPtr[0] - 0.5f) * stride + 0.5f;
        float y = (outputPtr[1] - 0.5f) * stride + 0.5f;
        float w = outputPtr[2] * inputWidth;
        float h = outputPtr[3] * inputHeight;

        // 转换为 OpenCV 的 Rect 格式，方便后面绘制
        boxes.push_back(cv::Rect(x, y, w, h));
        classIds.push_back(classIdx);
        scores.push_back(conf);
    }

    // 使用 NMS 去除重叠框
    std::vector<int> keep = nms(boxes, scores, iouThreshold);
    std::vector<cv::Rect> nmsBoxes;
    std::vector<int> nmsClassIds;
    std::vector<float> nmsScores;

    for (int idx : keep) {
        nmsBoxes.push_back(boxes[idx]);
        nmsClassIds.push_back(classIds[idx]);
        nmsScores.push_back(scores[idx]);
    }

    boxes = nmsBoxes;
    classIds = nmsClassIds;
    scores = nmsScores;
}

// 可视化检测结果
void visualizeDetection(const cv::Mat& image, const std::vector<cv::Rect>& boxes, 
                        const std::vector<int>& classIds, const std::vector<float>& scores, 
                        const std::vector<std::string>& classNames, const std::string& outputFileName) {
    cv::Mat outputImage = image.clone();

    // 绘制检测框
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(outputImage, boxes[i], cv::Scalar(0, 255, 0), 2);  // 绿色框
        std::string label = classNames[classIds[i]] + ": " + std::to_string(scores[i]);
        cv::putText(outputImage, label, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
    }

    // 保存可视化结果
    cv::imwrite(outputFileName, outputImage);
}

int main() {
    // 创建 TensorRT 推理引擎
    tensorrt::TensorRTInfer::Params params;
    params.onnxModelPath = "/home/rton/CppProj/TensorProj/test_bin/yolo11s.onnx";  // YOLOv8 的 ONNX 模型路径
    params.inputTensorNames = {"input"};   // 输入张量名称
    params.outputTensorNames = {"output"}; // 输出张量名称
    tensorrt::TensorRTInfer inferEngine(params);

    // 构建引擎
    if (!inferEngine.buildEngine()) {
        std::cerr << "Failed to build TensorRT engine." << std::endl;
        return -1;
    }

    // 加载类名
    std::vector<std::string> classNames = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
                                            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                                            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                                            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
                                            "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
                                            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
                                            "toothbrush" }; // 假设你有一个 80 类的标签

    // 读取输入图像
    std::string inputImageFile = "/home/rton/CppProj/TensorProj/test_bin/bus.jpg";
    cv::Mat image = cv::imread(inputImageFile);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    int inputWidth = 640;
    int inputHeight = 640;

    // 图像预处理：调整大小并归一化
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(inputWidth, inputHeight));
    resizedImage.convertTo(resizedImage, CV_32F, 1.0f / 255.0f);  // 归一化到 [0,1]

    // 将图像转换为一维数组
    std::vector<float> inputData(inputWidth * inputHeight * 3);
    std::memcpy(inputData.data(), resizedImage.data, inputData.size() * sizeof(float));

    // 设置输入数据并进行推理
    if (!inferEngine.setInputData(inputData)) {
        std::cerr << "Failed to set input data." << std::endl;
        return -1;
    }

    if (!inferEngine.infer()) {
        std::cerr << "Inference failed." << std::endl;
        return -1;
    }

    // 获取输出数据并进行后处理
    auto outputData = inferEngine.getOutputData();

    // 存储检测框、类别和置信度
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> scores;

    postprocessYOLOv8(outputData, inputWidth, inputHeight, 0.5f, 0.4f, boxes, classIds, scores);

    // 可视化检测结果并保存输出
    visualizeDetection(image, boxes, classIds, scores, classNames, "output.jpg");

    std::cout << "Detection completed. Results saved as 'output.jpg'." << std::endl;

    return 0;
}
