/**
 * @file onnx2engine_int8.cpp
 * @brief Convert ONNX model to TensorRT engine with INT8 quantization.
 *
 * Usage: ./onnx2engine_int8 <onnx_model> <engine_output> <calibration_data_dir> [batch_size] [calibration_table]
 *
 * This tool performs INT8 post-training quantization (PTQ) using TensorRT's
 * IInt8EntropyCalibrator2. It requires a set of calibration images to collect
 * activation statistics for optimal quantization.
 *
 * Supported image formats: .jpg, .jpeg, .png, .bmp, .ppm, .pgm (via OpenCV)
 */

#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <opencv2/opencv.hpp>

// ============================================================================
// Logger
// ============================================================================
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cout << "[";
            switch (severity)
            {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: std::cout << "INTERNAL_ERROR"; break;
                case nvinfer1::ILogger::Severity::kERROR:          std::cout << "ERROR"; break;
                case nvinfer1::ILogger::Severity::kWARNING:        std::cout << "WARNING"; break;
                case nvinfer1::ILogger::Severity::kINFO:           std::cout << "INFO"; break;
                case nvinfer1::ILogger::Severity::kVERBOSE:        std::cout << "VERBOSE"; break;
                default: break;
            }
            std::cout << "] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

// ============================================================================
// INT8 Calibrator - using IInt8EntropyCalibrator2 (recommended for most tasks)
// ============================================================================
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    /**
     * @brief Construct the calibrator.
     * @param calibrationDataDir  Directory containing calibration images.
     * @param calibrationTablePath  Path to cache/read calibration table.
     * @param inputName  Name of the network input tensor.
     * @param inputH  Input height expected by the network.
     * @param inputW  Input width expected by the network.
     * @param batchSize  Number of images per calibration batch.
     * @param inputC  Number of input channels (default 3 for RGB).
     */
    Int8EntropyCalibrator(const std::string& calibrationDataDir,
                          const std::string& calibrationTablePath,
                          const std::string& inputName,
                          int inputH, int inputW,
                          int batchSize = 1,
                          int inputC = 3)
        : mCalibrationTablePath(calibrationTablePath)
        , mInputName(inputName)
        , mInputH(inputH)
        , mInputW(inputW)
        , mInputC(inputC)
        , mBatchSize(batchSize)
        , mCurrentIndex(0)
    {
        // Load calibration image file list
        mImagePaths = listImages(calibrationDataDir);
        if (mImagePaths.empty())
        {
            std::cerr << "[Calibrator] WARNING: No calibration images found in "
                      << calibrationDataDir << std::endl;
        }

        // Allocate device memory for one batch
        size_t inputSize = mBatchSize * mInputC * mInputH * mInputW * sizeof(float);
        cudaMalloc(&mDeviceInput, inputSize);
    }

    ~Int8EntropyCalibrator() override
    {
        if (mDeviceInput)
        {
            cudaFree(mDeviceInput);
        }
    }

    int getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        if (mCurrentIndex + mBatchSize > static_cast<int>(mImagePaths.size()))
        {
            return false;  // No more batches
        }

        // Prepare one batch of pre-processed images
        size_t singleImageSize = mInputC * mInputH * mInputW * sizeof(float);
        std::vector<float> batchData(mBatchSize * mInputC * mInputH * mInputW);

        for (int b = 0; b < mBatchSize; ++b)
        {
            const std::string& imgPath = mImagePaths[mCurrentIndex + b];
            cv::Mat img = cv::imread(imgPath);
            if (img.empty())
            {
                std::cerr << "[Calibrator] WARNING: Failed to read image: " << imgPath << std::endl;
                std::fill(batchData.begin() + b * mInputC * mInputH * mInputW,
                          batchData.begin() + (b + 1) * mInputC * mInputH * mInputW,
                          0.0f);
                continue;
            }

            // Letterbox resize (matches YOLO26 inference preprocessing)
            cv::Mat letterboxed = letterbox(img, cv::Size(mInputW, mInputH));

            // Convert to float32 and normalize to [0, 1] only (no mean/std)
            letterboxed.convertTo(letterboxed, CV_32F, 1.0f / 255.0f);

            // HWC to CHW format (keep BGR channel order as-is)
            float* dst = batchData.data() + b * mInputC * mInputH * mInputW;
            int imgSize = mInputH * mInputW;
            std::vector<cv::Mat> channels;
            cv::split(letterboxed, channels);
            for (int c = 0; c < mInputC; ++c)
            {
                memcpy(dst + c * imgSize, channels[c].data, imgSize * sizeof(float));
            }
        }

        // Copy batch to GPU
        cudaMemcpy(mDeviceInput, batchData.data(),
                   mBatchSize * mInputC * mInputH * mInputW * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Find the correct binding index by name
        for (int i = 0; i < nbBindings; ++i)
        {
            if (std::strcmp(names[i], mInputName.c_str()) == 0)
            {
                bindings[i] = mDeviceInput;
                break;
            }
        }

        mCurrentIndex += mBatchSize;
        std::cout << "[Calibrator] Calibration batch "
                  << (mCurrentIndex / mBatchSize) << " / "
                  << (mImagePaths.size() / mBatchSize) << std::endl;

        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        mCalibrationCache.clear();
        std::ifstream cacheFile(mCalibrationTablePath, std::ios::binary);
        if (cacheFile)
        {
            cacheFile.seekg(0, std::ios::end);
            length = cacheFile.tellg();
            cacheFile.seekg(0, std::ios::beg);
            mCalibrationCache.resize(length);
            cacheFile.read(mCalibrationCache.data(), length);
            std::cout << "[Calibrator] Loaded calibration cache from: "
                      << mCalibrationTablePath << " (" << length << " bytes)" << std::endl;
            return mCalibrationCache.data();
        }
        length = 0;
        return nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        std::ofstream cacheFile(mCalibrationTablePath, std::ios::binary);
        if (cacheFile)
        {
            cacheFile.write(static_cast<const char*>(cache), length);
            std::cout << "[Calibrator] Calibration cache written to: "
                      << mCalibrationTablePath << " (" << length << " bytes)" << std::endl;
        }
        else
        {
            std::cerr << "[Calibrator] WARNING: Failed to write calibration cache to: "
                      << mCalibrationTablePath << std::endl;
        }
    }

private:
    /**
     * @brief Letterbox resize - maintains aspect ratio with padding.
     * Matches YOLO26 inference preprocessing.
     */
    static cv::Mat letterbox(const cv::Mat& image, const cv::Size& new_shape,
                             const cv::Scalar& color = cv::Scalar(114, 114, 114))
    {
        float width_ratio = new_shape.width / static_cast<float>(image.cols);
        float height_ratio = new_shape.height / static_cast<float>(image.rows);
        float scale = std::min(width_ratio, height_ratio);

        int new_width = static_cast<int>(image.cols * scale);
        int new_height = static_cast<int>(image.rows * scale);

        cv::Mat resized_img;
        cv::resize(image, resized_img, cv::Size(new_width, new_height));

        cv::Mat canvas = cv::Mat::zeros(new_shape.height, new_shape.width, image.type());
        canvas.setTo(color);

        int top_padding = (new_shape.height - new_height) / 2;
        int left_padding = (new_shape.width - new_width) / 2;

        cv::Rect roi(left_padding, top_padding, new_width, new_height);
        resized_img.copyTo(canvas(roi));

        return canvas;
    }

    /**
     * @brief List all supported image files in a directory.
     */
    static std::vector<std::string> listImages(const std::string& dirPath)
    {
        std::vector<std::string> result;
        DIR* dir = opendir(dirPath.c_str());
        if (!dir)
        {
            std::cerr << "[Calibrator] WARNING: Cannot open directory: " << dirPath << std::endl;
            return result;
        }

        // Supported extensions
        static const std::vector<std::string> exts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm"
        };

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string name = entry->d_name;
            std::string ext = name.substr(name.find_last_of('.') != std::string::npos
                                              ? name.find_last_of('.')
                                              : 0);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(exts.begin(), exts.end(), ext) != exts.end())
            {
                result.push_back(dirPath + "/" + name);
            }
        }
        closedir(dir);

        // Sort for reproducibility
        std::sort(result.begin(), result.end());
        return result;
    }

    std::string mCalibrationTablePath;
    std::string mInputName;
    int mInputH;
    int mInputW;
    int mInputC;
    int mBatchSize;
    int mCurrentIndex;
    void* mDeviceInput = nullptr;
    std::vector<std::string> mImagePaths;
    std::vector<char> mCalibrationCache;
};

// ============================================================================
// Helper: query network input dimensions
// ============================================================================
static void queryInputInfo(nvinfer1::INetworkDefinition* network,
                           std::string& inputName,
                           int& inputH, int& inputW, int& inputC)
{
    inputName = "";
    inputH = 0;
    inputW = 0;
    inputC = 0;

    int nbInputs = network->getNbInputs();
    if (nbInputs == 0)
    {
        std::cerr << "Network has no inputs!" << std::endl;
        return;
    }

    nvinfer1::ITensor* input = network->getInput(0);
    inputName = input->getName();

    nvinfer1::Dims dims = input->getDimensions();
    // For NCHW format: dims = [N, C, H, W]
    if (dims.nbDims >= 3)
    {
        inputC = dims.d[dims.nbDims - 3];
        inputH = dims.d[dims.nbDims - 2];
        inputW = dims.d[dims.nbDims - 1];
    }
    else if (dims.nbDims >= 2)
    {
        inputC = 1;
        inputH = dims.d[dims.nbDims - 2];
        inputW = dims.d[dims.nbDims - 1];
    }

    std::cout << "Network input: name=\"" << inputName
              << "\", dims=[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  => C=" << inputC << ", H=" << inputH << ", W=" << inputW << std::endl;
}

// ============================================================================
// Main conversion function
// ============================================================================
void convertOnnxToInt8Engine(const std::string& onnxModelPath,
                             const std::string& enginePath,
                             const std::string& calibrationDataDir,
                             int batchSize,
                             const std::string& calibrationTablePath)
{
    // 1. Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return;
    }

    // 2. Create network with explicit batch
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        return;
    }

    // 3. Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cerr << "Failed to create builder config" << std::endl;
        return;
    }

    // 4. Create ONNX parser and parse model
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return;
    }

    if (!parser->parseFromFile(onnxModelPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "Failed to parse ONNX file: " << onnxModelPath << std::endl;
        return;
    }
    std::cout << "Successfully parsed ONNX model: " << onnxModelPath << std::endl;

    // 5. Query network input info
    std::string inputName;
    int inputH, inputW, inputC;
    queryInputInfo(network.get(), inputName, inputH, inputW, inputC);

    if (inputH <= 0 || inputW <= 0)
    {
        std::cerr << "Could not determine network input dimensions. "
                     "Please ensure the ONNX model has fixed input shapes." << std::endl;
        return;
    }

    // 6. Set network input dimensions for multi-batch support
    // For static-shape ONNX models, directly set the input tensor dimensions
    if (batchSize > 1)
    {
        nvinfer1::ITensor* inputTensor = network->getInput(0);
        nvinfer1::Dims4 newDims(batchSize, inputC, inputH, inputW);
        inputTensor->setDimensions(newDims);
        std::cout << "Network input dimensions set to: [" << batchSize << ", " << inputC
                  << ", " << inputH << ", " << inputW << "]" << std::endl;
    }

    // 7. Set workspace size
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1 GB

    // 8. Enable INT8
    std::cout << "Setting INT8 mode..." << std::endl;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    // Also enable FP16 as a fallback for layers that don't support INT8
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 fallback enabled (platform has fast FP16)." << std::endl;
    }

    // 9. Create and set INT8 calibrator
    auto calibrator = std::make_unique<Int8EntropyCalibrator>(
        calibrationDataDir, calibrationTablePath,
        inputName, inputH, inputW, batchSize, inputC > 0 ? inputC : 3);
    config->setInt8Calibrator(calibrator.get());

    // 10. Build the engine
    std::cout << "Building INT8 engine (this may take a while)..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cerr << "Failed to build INT8 engine!" << std::endl;
        return;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Engine built successfully in " << elapsed << " seconds." << std::endl;

    // 11. Write serialized engine to file
    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Failed to open engine file for writing: " << enginePath << std::endl;
        return;
    }

    engineFile.write(static_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    std::cout << "INT8 TensorRT engine saved to: " << enginePath << std::endl;
    std::cout << "Engine size: " << (plan->size() >> 20) << " MB" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName
              << " <onnx_model_path> <engine_output_path> <calibration_data_dir>"
              << " [batch_size] [calibration_table_path]"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  onnx_model_path          Path to the input ONNX model file" << std::endl;
    std::cout << "  engine_output_path       Path for the output TensorRT engine file" << std::endl;
    std::cout << "  calibration_data_dir     Directory containing calibration images"
              << " (jpg/png/bmp/ppm/pgm)" << std::endl;
    std::cout << "  batch_size               Batch size for calibration (default: 1)" << std::endl;
    std::cout << "  calibration_table_path   Path for calibration cache file"
              << " (default: calibration.table)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << progName
              << " model.onnx model_int8.engine ./calibration_images 1 calib.table"
              << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        printUsage(argv[0]);
        return -1;
    }

    std::string onnxModelPath = argv[1];
    std::string enginePath = argv[2];
    std::string calibrationDataDir = argv[3];
    int batchSize = (argc > 4) ? std::atoi(argv[4]) : 1;
    std::string calibrationTablePath = (argc > 5) ? argv[5] : "calibration.table";

    if (batchSize <= 0)
    {
        std::cerr << "Invalid batch size: " << batchSize << std::endl;
        return -1;
    }

    std::cout << "=== ONNX to INT8 TensorRT Engine Converter ===" << std::endl;
    std::cout << "ONNX model:           " << onnxModelPath << std::endl;
    std::cout << "Engine output:         " << enginePath << std::endl;
    std::cout << "Calibration data dir:  " << calibrationDataDir << std::endl;
    std::cout << "Batch size:            " << batchSize << std::endl;
    std::cout << "Calibration table:     " << calibrationTablePath << std::endl;
    std::cout << "===============================================" << std::endl;

    convertOnnxToInt8Engine(onnxModelPath, enginePath, calibrationDataDir,
                            batchSize, calibrationTablePath);

    return 0;
}