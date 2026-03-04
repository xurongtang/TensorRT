# TensorRTx MLP Demo

本项目基于 [tensorrtx](https://github.com/wang-xinyu/tensorrtx) 的 `mlp` 示例（`trt10` 分支），展示了如何将 PyTorch 定义的 MLP 模型转换为 TensorRT 引擎并进行推理。包含基础示例和进阶扩展示例。

---

## 📦 项目结构

```
.
├── mlp.py               # 基础示例：定义并导出 MLP 模型为 .wts
├── mlp.cpp              # 基础示例：加载 .wts 构建 TensorRT 引擎并推理
├── mlp.wts              # 基础示例生成的权重文件
├── mlp_plus.py          # 扩展示例：更复杂的 MLP 模型定义与导出
├── mlp_plus.cpp         # 扩展示例：对应的 TensorRT 推理代码
├── mlp_plus.wts         # 扩展示例生成的权重文件
├── build.sh             # 一键编译脚本
└── CMakeLists.txt       # CMake 编译配置文件
```

---

## 🚀 快速开始

### 1. 基础示例

#### 1.1 生成权重文件（optional）（.wts）

运行 Python 脚本生成权重文件：

```bash
python mlp.py
```

将会在当前目录下生成 `mlp.wts` 文件。

#### 1.2 编译与运行

执行编译及测试脚本：

```bash
./build.sh
```

### 2. 扩展示例（mlp_plus）

#### 2.1 生成权重文件

```bash
python mlp_plus.py
```

生成 `mlp_plus.wts`。

#### 2.2 修改编译目标

编辑 `CMakeLists.txt`，将：

```cmake
add_executable(mlp mlp_demo.cpp)
```

修改为：

```cmake
add_executable(mlp mlp_plus.cpp)
```

#### 2.3 编译与运行

```bash
./build.sh
```

---

## ⚙️ 依赖环境

- CUDA >= 11.0
- TensorRT >= 8.0 / 10.0（对应 `trt10` 分支）
- PyTorch >= 1.8（用于导出 `.wts`）
- CMake >= 3.10
- g++ 支持 C++17

---

## 📝 注意事项

- 请确保 TensorRT 库路径已正确配置在 `CMakeLists.txt` 中。

---

## 🙏 致谢

感谢 [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) 项目提供的 TensorRT 示例。

---