# TensorRT Installation, Deployment, and Usage Guide

> 📅 Last Updated: 2026-02-22  
> 🌐 [English Version](README_en.md) | [中文版](README.md)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)

---

## 📋 Project Overview

This document records the complete process of deploying NVIDIA TensorRT 10.11.0.33 under the **Windows 11 + WSL2 + Ubuntu 22.04** environment, including environment configuration, compilation verification, and sample program execution. It is suitable for CUDA-accelerated inference development on RTX 40/50 series graphics cards.

---

## 🔧 System Requirements

### Hardware Configuration
| Component | Requirements/Version |
|-----------|---------------------|
| Operating System | Windows 11 + WSL2 + Ubuntu 22.04 LTS |
| GPU | NVIDIA RTX 5060 Ti |

### Software Dependencies
| Component | Version | Verification Command |
|-----------|---------|----------------------|
| NVIDIA Driver | 580.91 | `nvidia-smi` |
| CUDA Toolkit | 12.6 (development) / 12.9 (TensorRT compatible) | `nvcc -V` |
| TensorRT | 10.11.0.33_cuda12.9 | `python3 -c "import tensorrt; print(tensorrt.__version__)"` |
| CMake | ≥ 3.20 | `cmake --version` |
| GCC/G++ | 9~12 (recommended 11) | `gcc --version` |

> ✅ Verify WSL2 GPU support is enabled:
> ```bash
> nvidia-smi  # Should display GPU information normally
> ```

---

## 🚀 Quick Start

Replace `~/CppProj/TensorProj/` with the actual path

```bash
# 1. Download and extract TensorRT
tar -xzvf TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz -C ~/CppProj/TensorProj/

# 2. Temporarily configure environment variables (valid for current terminal)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/CppProj/TensorProj/TensorRT-10.11.0.33/lib

# 3. Compile sample programs
cd ~/CppProj/TensorProj/install_check
./build.sh

# 4. Run tests
./sample_onnx_mnist --datadir ./data/mnist/
```

For detailed compilation information, please read: `install_check/demo/CMakeLists.txt` and `install_check/build.sh` files

**Reference**
- 文档: https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
- 下载：https://developer.nvidia.com/tensorrt/download/10x  
- 测试程序：https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST  
- 数据：https://github.com/NVIDIA/TensorRT/releases  