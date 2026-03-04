#include <chrono>     // for timing the execution
#include <fstream>    // for file-handling
#include <map>        // for weight maps
#include "NvInfer.h"  // TensorRT library
#include "iostream"   // Standard input/output library
#include "logging.h"  // logging file -- by NVIDIA

// provided by nvidia for using TensorRT APIs
using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 1;

/** ////////////////////////////
// DEPLOYMENT RELATED /////////
////////////////////////////*/

// Weights为 TensorRT 中的 nvinfer1::Weights 结构体。
// 其主要包括三个字段：type（权重的数据类型：浮点、整形等）、values（）、count(权重元素总数)
//type：权重的数据类型，这里是 DataType::kFLOAT，表示 32 位浮点数（也可以是 kHALF、kINT8 等）。
// values：指向实际权重数据的指针（void const*），这里通过 malloc 分配内存并读取 .wts 文件中的十六进制数值填充。
// const void* 或 void const*：表示这个指针所指向的内容是只读的，你不能通过该指针修改它所指向的内存。
// count：权重元素的总数（例如卷积层的权重数量 = 输出通道 × 输入通道 × 高 × 宽）。

std::map<std::string, Weights> loadWeights(const std::string file) {
    /**
     * Parse the .wts file and store weights in dict format.
     *
     * @param file path to .wts file
     * @return weight_map: dictionary containing weights and their values
     */

    std::cout << "[INFO]: Loading weights..." << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight file
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read number of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // Loop through number of line, actually the number of weights & biases
    while (count--) {
        // TensorRT weights
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of weights
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // 分配一块足够存储 size 个 uint32_t 整数的内存，用于从 .wts 文件中读取十六进制权重值。
        // reinterpret_cast<uint32_t*>(...)
        // 这是 C++ 的强制类型转换操作符，用于直接重新解释指针的位模式。
        // 它告诉编译器：“把这块内存当作 uint32_t* 来看待，不要做任何检查或转换”。
        // 这里将 malloc 返回的 void* 转换为 uint32_t*，以便后续通过 val[x] 访问。
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            // Change hex values to uint32 (for higher values)
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;

        // Add weight values against its name (key)
        weightMap[w_name] = wt;
    }
    return weightMap;
}

ICudaEngine* createMLPEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    /**
     * Create Multi-Layer Perceptron using the TRT Builder and Configurations
     *
     * @param maxBatchSize: batch size for built TRT model
     * @param builder: to build engine and networks
     * @param config: configuration related to Hardware
     * @param dt: datatype for model layers
     * @return engine: TRT model
     */

    // 构建的计算方式定义了模型的计算图
    // 是的，TensorRT（TRT）确实是通过你在 C++ 中调用 API 的顺序来隐式构建计算图（Computational Graph）的。
    // 层的输入决定了依赖关系
    // addMatrixMultiply(A, B) 表示 A × B
    // 这里 A = weightLayer->getOutput(0)，B = data
    // TensorRT 自动知道：必须先计算 weightLayer 和 data，才能执行矩阵乘法
    // 📌 不是靠代码书写顺序，而是靠张量（ITensor）的引用关系！
    // 即使你把 biasLayer 的创建写在 mmLayer 前面，只要它只被用于后面的 addElementWise，拓扑结构就不会变。
    // 调用：builder->buildEngineWithConfig(*network, *config); TensorRT 会：
    // 遍历所有层和张量
    // 根据输入/输出依赖关系构建完整的 DAG
    // 进行拓扑排序，确定实际执行顺序
    // 应用优化（层融合、内存复用、精度转换等）
    // 所以，你不需要手动排好顺序，只要正确连接张量即可。

    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("../mlp.wts");

    // Create an empty network
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input
    ITensor* data = network->addInput("data", dt, Dims2{1, INPUT_SIZE});
    assert(data);

    // Create constant tensor for weights
    Dims weightDims = Dims2{OUTPUT_SIZE, INPUT_SIZE};
    Weights w = weightMap["linear.weight"];
    IConstantLayer* weightLayer = network->addConstant(weightDims, w);
    assert(weightLayer);

    // Matrix multiply: Wx
    IMatrixMultiplyLayer* mmLayer = network->addMatrixMultiply(*weightLayer->getOutput(0), MatrixOperation::kNONE,
                                                               *data, MatrixOperation::kNONE);
    assert(mmLayer);

    // Add bias
    Dims biasDims = Dims2{OUTPUT_SIZE, 1};
    Weights b = weightMap["linear.bias"];
    IConstantLayer* biasLayer = network->addConstant(biasDims, b);
    assert(biasLayer);

    IElementWiseLayer* outLayer =
            network->addElementWise(*mmLayer->getOutput(0), *biasLayer->getOutput(0), ElementWiseOperation::kSUM);
    assert(outLayer);

    // Set output
    outLayer->getOutput(0)->setName("out");
    network->markOutput(*outLayer->getOutput(0));

    // Build engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    // Clean up
    for (auto& mem : weightMap)
        free((void*)(mem.second.values));

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    /**
     * Create engine using TensorRT APIs
     *
     * @param maxBatchSize: for the deployed model configs
     * @param modelStream: shared memory to store serialized model
     */

    // Create builder with the help of logger
    IBuilder* builder = createInferBuilder(gLogger);

    // Create hardware configs
    IBuilderConfig* config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine* engine = createMLPEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();
}

void performSerialization() {
    /**
     * Serialization Function
     */
    // Shared memory object
    IHostMemory* modelStream{nullptr};

    // Write model into stream
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);

    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // Open the file and write the contents there in binary format
    std::ofstream p("../mlp.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./mlp -d`" << std::endl;
}

/** ////////////////////////////
// INFERENCE RELATED //////////
////////////////////////////*/
void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    /**
     * Perform inference using the CUDA context
     *
     * @param context: context created by engine
     * @param input: input from the host
     * @param output: output to save on host
     * @param batchSize: batch size for TRT model
     */

    // Get engine from the context
    const ICudaEngine& engine = context.getEngine();

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
    std::ifstream file("../mlp.engine", std::ios::binary);
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
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char-stream
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    delete[] trtModelStream;

    // create execution context -- required for inference executions
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // input and output
    float data[1] = {12.0f};
    float out[1] = {0.0f};

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
    std::cout << "\nOutput:\t" << out[0] << std::endl;
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
