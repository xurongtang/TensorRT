# TensorRT 推理实践与学习项目

本项目旨在提供一套完整的 TensorRT 学习与推理实践框架。涵盖了从环境检查、底层 API 使用、模型转换、推理基类封装到具体目标检测模型（YOLO）的端到端部署流程。

## 📁 目录结构

```text
.
├── 3rdparty                 # 第三方依赖库
├── install_check            # TensorRT 环境安装检查脚本
├── mlp_demo                 # MLP 手写计算图 Demo（学习 TRT API）
├── onnx2tensorrt            # ONNX 模型转 TensorRT Engine 脚本
├── tensorrt_inference_base  # TensorRT 推理基类（封装加载与推理）
├── yolom_26_trt             # YOLO 目标检测 TensorRT 推理实现
└── README.md                # 项目说明文档
```

## 📂 模块说明

### 1. `install_check` (环境检查)
用于验证当前环境中 TensorRT 是否正确安装及版本信息。
- **功能**：检查 CUDA、cuDNN 及 TensorRT 库的链接情况。
- **用途**：在开始编译或运行其他模块前，建议先运行此模块确保环境无误。

### 2. `mlp_demo` (API 学习, reference by: [tensorrtx](https://github.com/wang-xinyu/tensorrtx))
通过手动构建多层感知机（MLP）计算图，深入学习 TensorRT 底层 API。
- **功能**：
  - 使用 TensorRT API 手动定义网络层（Network Definition）。
  - 构建推理引擎（Build Engine）。
  - 执行推理并验证结果。
- **用途**：适合初学者理解 TensorRT 的 `Builder`, `Network`, `Engine`, `Context` 核心概念。


### 3. `onnx2tensorrt` (模型转换)
提供将 ONNX 格式模型转换为 TensorRT 引擎（.engine / .plan）的工具脚本。
- **功能**：加载 ONNX 模型，指定精度（TODO），生成序列化引擎文件。
- **用途**：模型部署前的标准转换步骤。

### 4. `tensorrt_inference_base` (推理基类)
封装了 TensorRT 推理的通用流程，避免重复造轮子。
- **功能**：
  - 统一封装 Engine 的加载与反序列化。
  - 封装内存分配、数据拷贝、推理执行及资源释放。
  - 提供虚接口供具体模型继承实现预处理和后处理。
- **用途**：作为其他具体推理项目（如 YOLO）的基座，提高代码复用性。

### 5. `yolom_26_trt` (YOLO26 目标检测)
基于 `tensorrt_inference_base` 实现的具体业务案例。
- **功能**：完成 YOLO 系列模型的目标检测推理。
- **包含**：图像预处理、推理执行、后处理（NMS）、结果绘制。
- **用途**：端到端的部署参考示例。



## 🔧 依赖环境
| 组件 | 版本 | 验证命令 |
|------|------|----------|
| NVIDIA Driver | 580.91 | `nvidia-smi` |
| CUDA Toolkit | 12.6（开发）/ 12.9（TensorRT 匹配） | `nvcc -V` |
| TensorRT | 10.11.0.33_cuda12.9 | `python3 -c "import tensorrt; print(tensorrt.__version__)"` |
| CMake | ≥ 3.20 | `cmake --version` |
| GCC/G++ | 9~12（推荐 11） | `gcc --version` |