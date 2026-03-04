/*
    Author: xurongtang
    Reference: https://github.com/wang-xinyu/tensorrtx/mlp 
    MLP Demo 
    仿照现有的示例，构建一个多层的MLP
    输入为1x2x5维度的数据：
    [[1,1,1,1,1],
     [2,2,2,2,2]]

    层数包括三层。Linear(5,10), ReLU, Linear(10,10), ReLU, Linear(10,1), Sigmoid
    其中层数的权重均为0.1, bias都为0.2：
    计算示例：
        第一层输出（Linear + ReLU）：
        线性: x @ W1 + b1
            样本1: 1×0.1×5 + 0.2 = 0.7  → 每个神经元均为 0.7，shape [1,10]
            样本2: 2×0.1×5 + 0.2 = 1.2  → 每个神经元均为 1.2，shape [1,10]
        ReLU后（均为正数不变）：
            样本1: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
            样本2: [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

        第二层输出（Linear + ReLU）：
        线性: x @ W2 + b2
            样本1: 0.7×0.1×10 + 0.2 = 0.9  → 每个神经元均为 0.9，shape [1,10]
            样本2: 1.2×0.1×10 + 0.2 = 1.4  → 每个神经元均为 1.4，shape [1,10]
        ReLU后（均为正数不变）：
            样本1: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
            样本2: [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]

        第三层输出（Linear + Sigmoid）：
        线性: x @ W3 + b3
            样本1: 0.9×0.1×10 + 0.2 = 1.1
            样本2: 1.4×0.1×10 + 0.2 = 1.6
        Sigmoid后：
            样本1: sigmoid(1.1) ≈ 0.7503
            样本2: sigmoid(1.6) ≈ 0.8320

        最终输出（shape [2,1]）：
                [[0.7503],
                [0.8320]]
*/

#include <chrono>
#include <fstream>
#include <map>
#include <NvInfer.h>
#include "iostream"
#include "logging.h"

static Logger gLogger;

const int INPUT_SIZE = 10;
const int OUTPUT_SIZE = 2;

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) { 
    std::cout << "Loading weights: " << file << std::endl;
    // 构建map
    std::map<std::string, nvinfer1::Weights> weightMap;

    // 读取文件
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // 读取层总数
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // 逐层读取权重信息
    while (count--) { 
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;
        // 遍历读取所有权重
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0; x < size; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[w_name] = wt;
    }
    return weightMap;
}

// 构建计算图
nvinfer1::ICudaEngine* createMLPEngine(
    unsigned int MaxBatchSize,
    nvinfer1::IBuilder* builder,
    nvinfer1::IBuilderConfig* config,
    nvinfer1::DataType dt
){
    /*
     * 创建计算图
     *
     * @param maxBatchSize: 创建模型的最大batch size
     * @param builder: 去创建的TRT推理引擎
     * @param config: 硬件相关参数
     * @param dt: 模型的数据类型
     * @return engine: TRT model

    */

    // 首先加载权重
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights("../mlp_plus.wts");

    // 创建一个空网络对象
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    // 创建输入层
    nvinfer1::ITensor* data = network->addInput("data", dt, nvinfer1::Dims3{1, 2, 5});

    nvinfer1::IShuffleLayer* reshape = network->addShuffle(*data);
    reshape->setReshapeDimensions(nvinfer1::Dims2{2, 5});
    nvinfer1::ITensor* data_flat = reshape->getOutput(0);

    // 创建权重层
    // 创建权重维度
    nvinfer1::Dims fisrt_layer_w_dims = nvinfer1::Dims2{5, 10};
    nvinfer1::Weights first_layer_w = weightMap["layer1.weight"];
    // 将权重写入网络中
    nvinfer1::IConstantLayer* first_layer = network->addConstant(fisrt_layer_w_dims, first_layer_w);

    // 定义输入层和权重层之间的乘法
    nvinfer1::IMatrixMultiplyLayer* first_layer_mul = network->addMatrixMultiply(
        *data_flat,                 nvinfer1::MatrixOperation::kNONE,
        *first_layer->getOutput(0), nvinfer1::MatrixOperation::kNONE
    );

    // 读取偏置层权重
    nvinfer1::Dims fisrt_layer_b_dims = nvinfer1::Dims2{1, 10};
    nvinfer1::Weights first_layer_b_value = weightMap["layer1.bias"];

    // 创建偏执层
    nvinfer1::IConstantLayer* first_layer_b = network->addConstant(fisrt_layer_b_dims, first_layer_b_value);

    // 定义偏置层和乘法层之间的加法
    nvinfer1::IElementWiseLayer* first_layer_add = network->addElementWise(
        *first_layer_b->getOutput(0), 
        *first_layer_mul->getOutput(0), 
        nvinfer1::ElementWiseOperation::kSUM
    );

    // 创建激活层
    nvinfer1::IActivationLayer* first_layer_relu = network->addActivation(
        *first_layer_add->getOutput(0), nvinfer1::ActivationType::kRELU
    );

    // 读取第二层权重
    nvinfer1::Dims second_layer_w_dims = nvinfer1::Dims2{10, 10};
    nvinfer1::Weights second_layer_w = weightMap["layer2.weight"];
    nvinfer1::IConstantLayer* second_layer = network->addConstant(second_layer_w_dims, second_layer_w);

    // 创建第二层乘法层
    nvinfer1::IMatrixMultiplyLayer* second_layer_mul = network->addMatrixMultiply(
        *first_layer_relu->getOutput(0),  nvinfer1::MatrixOperation::kNONE,
        *second_layer->getOutput(0),      nvinfer1::MatrixOperation::kNONE
    );

    // 创建第二层偏置层（维度、权重、创建层、定义计算方法）
    nvinfer1::Dims second_layer_b_dims = nvinfer1::Dims2{1, 10};
    nvinfer1::Weights second_layer_b_value = weightMap["layer2.bias"];
    nvinfer1::IConstantLayer* second_layer_b = network->addConstant(second_layer_b_dims, second_layer_b_value);
    nvinfer1::IElementWiseLayer* second_layer_add = network->addElementWise(
        *second_layer_mul->getOutput(0), 
        *second_layer_b->getOutput(0), 
        nvinfer1::ElementWiseOperation::kSUM
    );

    // 创建第二层激活层
    nvinfer1::IActivationLayer* second_layer_relu = network->addActivation(
        *second_layer_add->getOutput(0),
        nvinfer1::ActivationType::kRELU
    );

    // 创建第三层权重
    nvinfer1::Dims third_layer_w_dims = nvinfer1::Dims2{10, 1};
    nvinfer1::Dims output_dims = nvinfer1::Dims2{1, 1};
    nvinfer1::Weights third_layer_w = weightMap["layer3.weight"];
    nvinfer1::IConstantLayer* third_layer = network->addConstant(third_layer_w_dims, third_layer_w);
    nvinfer1::IMatrixMultiplyLayer* third_layer_mul = network->addMatrixMultiply(
        *second_layer_relu->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *third_layer->getOutput(0), nvinfer1::MatrixOperation::kNONE
    );

    // 创建第三层偏置层
    nvinfer1::Dims third_layer_b_dims = nvinfer1::Dims2{1, 1};
    nvinfer1::Weights third_layer_b_value = weightMap["layer3.bias"];
    nvinfer1::IConstantLayer* third_layer_b = network->addConstant(third_layer_b_dims, third_layer_b_value);
    nvinfer1::IElementWiseLayer* third_layer_add = network->addElementWise(
        *third_layer_mul->getOutput(0), *third_layer_b->getOutput(0), nvinfer1::ElementWiseOperation::kSUM
    );

    // 最终sigmoid
    nvinfer1::IActivationLayer* third_layer_sigmoid = network->addActivation(
        *third_layer_add->getOutput(0), nvinfer1::ActivationType::kSIGMOID
    );

    // 设置输出
    third_layer_sigmoid->getOutput(0)->setName("out");
    network->markOutput(*third_layer_sigmoid->getOutput(0));

    // 创建推理引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 清理权重所占用的内存
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, nvinfer1::IHostMemory** modelStream) {
    // Create builder with the help of logger
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // Create hardware configs
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // Build an engine
    nvinfer1::ICudaEngine* engine = createMLPEngine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();
}

void performSerialization() {
    /**
     * Serialization Function
     */
    // Shared memory object
    nvinfer1::IHostMemory* modelStream{nullptr};

    // Write model into stream
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);

    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // Open the file and write the contents there in binary format
    std::ofstream p("../mlp_plus.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./mlp -d`" << std::endl;
}

void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize) {
    /**
     * Perform inference using the CUDA context
     *
     * @param context: context created by engine
     * @param input: input from the host
     * @param output: output to save on host
     * @param batchSize: batch size for TRT model
     */

    // Get engine from the context
    const nvinfer1::ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbIOTensors() number of buffers.
    assert(engine.getNbIOTensors() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const char* inputName = "data";
    const char* outputName = "out";

    // Create GPU buffers on device -- allocate memory for input and output
    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};
    cudaMalloc(&deviceInput, batchSize * INPUT_SIZE * sizeof(float));
    cudaMalloc(&deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float));

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy input from host (CPU) to device (GPU)  in stream
    cudaMemcpyAsync(deviceInput, input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);

    // set Tensor address (TensorRT 10)
    context.setTensorAddress(inputName, deviceInput);
    context.setTensorAddress(outputName, deviceOutput);

    // execute inference using context provided by engine
    context.enqueueV3(stream);

    // copy output back from device (GPU) to host (CPU)
    cudaMemcpyAsync(output, deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // synchronize the stream to prevent issues
    cudaStreamSynchronize(stream);

    // Release stream and buffers (memory)
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void performInference() {
    /**
     * Get inference using the pre-trained model
     */
    // read model from the engine file
    std::ifstream file("../mlp_plus.engine", std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file!" << std::endl;
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // create a runtime (required for deserialization of model) with NVIDIA's logger
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char-stream
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    delete[] trtModelStream;

    // create execution context -- required for inference executions
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // input and output
    float data[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float out[2] = {0.0f, 0.0f};

    // time the execution
    auto start = std::chrono::system_clock::now();

    // do inference using the parameters
    doInference(*context, data, out, 1);

    // time the execution
    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // output result
    std::cout << "\nInput:\t" << data[0];
    std::cout << "\nOutput:\t";
    for (auto &i: out){
        std::cout << "\t" << i;
    }
    std::cout << std::endl;
}

int checkArgs(int argc, char** argv) {
    /**
     * Parse command line arguments
     *
     * @param argc: argument count
     * @param argv: arguments vector
     * @return int: a flag to perform operation
     */

    if (argc != 2) {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./mlp -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mlp -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    if (std::string(argv[1]) == "-s") {
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        return 2;
    }
    return -1;
}

int main(int argc, char** argv) {
    int args = checkArgs(argc, argv);
    if (args == 1)
        performSerialization();
    else if (args == 2)
        performInference();
    return 0;
}
