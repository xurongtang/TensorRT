# TensorRT 安装部署及使用指南

> 📅 最后更新：2026-02-22  
> 🌐 [English Version](README_en.md) | [中文版](README.md)

---

## 📌 目录

- [项目概述](#-项目概述)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)

---

## 📋 项目概述

本文档记录在 **Windows 11 + WSL2 + Ubuntu 22.04** 环境下，部署 NVIDIA TensorRT 10.11.0.33 的完整流程，包含环境配置、编译验证及示例程序运行，适用于 RTX 40/50 系列显卡的 CUDA 加速推理开发。

---

## 🔧 环境要求

### 硬件配置
| 组件 | 要求/版本 |
|------|----------|
| 操作系统 | Windows 11 + WSL2 + Ubuntu 22.04 LTS |
| GPU | NVIDIA RTX 5060 Ti |

### 软件依赖
| 组件 | 版本 | 验证命令 |
|------|------|----------|
| NVIDIA Driver | 580.91 | `nvidia-smi` |
| CUDA Toolkit | 12.6（开发）/ 12.9（TensorRT 匹配） | `nvcc -V` |
| TensorRT | 10.11.0.33_cuda12.9 | `python3 -c "import tensorrt; print(tensorrt.__version__)"` |
| CMake | ≥ 3.20 | `cmake --version` |
| GCC/G++ | 9~12（推荐 11） | `gcc --version` |

> ✅ 验证 WSL2 已启用 GPU 支持：
> ```bash
> nvidia-smi  # 应能正常显示 GPU 信息
> ```

---

## 🚀 快速开始

注意将 `~/CppProj/TensorProj/` 替换为实际路径

```bash
# 1. 下载并解压 TensorRT
tar -xzvf TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz -C ~/CppProj/TensorProj/

# 2. 临时配置环境变量（当前终端生效）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/CppProj/TensorProj/TensorRT-10.11.0.33/lib

# 3. 编译示例程序
cd ~/CppProj/TensorProj/install_check
./build.sh

# 4. 运行测试
./sample_onnx_mnist --datadir ./data/mnist/
```

详细编译细节请阅读：`install_check/demo/CMakeLists.txt`和`install_check/build.sh`文件

**Reference**
- 文档: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
- 下载：https://developer.nvidia.com/tensorrt/download/10x  
- 测试程序：https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST  
- 数据：https://github.com/NVIDIA/TensorRT/releases  