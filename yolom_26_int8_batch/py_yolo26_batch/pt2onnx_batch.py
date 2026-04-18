"""
将 yolo26m.pt 转换为 ONNX 格式（batch=4）

使用 ultralytics 的 export API 导出 ONNX 模型，
batch size 设置为 4，适配后续 TensorRT INT8 批量推理流程。

输出文件：yolo26m.onnx（保存在 quant_convert/ 目录下）
"""

from ultralytics import YOLO
import os
import shutil

# ============================================================================
# 配置
# ============================================================================
PT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26m.pt")
ONNX_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quant_convert")
ONNX_OUTPUT_NAME = "yolo26m.onnx"
BATCH_SIZE = 4
IMG_SIZE = 640

def main():
    print("=" * 60)
    print("YOLO26m PT → ONNX 转换 (batch={})".format(BATCH_SIZE))
    print("=" * 60)

    # 检查 pt 模型是否存在
    if not os.path.isfile(PT_MODEL_PATH):
        print(f"[ERROR] 模型文件不存在: {PT_MODEL_PATH}")
        return

    print(f"[INFO] PT 模型路径: {PT_MODEL_PATH}")
    print(f"[INFO] 输出目录:     {ONNX_OUTPUT_DIR}")
    print(f"[INFO] Batch Size:   {BATCH_SIZE}")
    print(f"[INFO] 图像尺寸:     {IMG_SIZE}x{IMG_SIZE}")
    print()

    # 加载模型
    print("[INFO] 加载模型...")
    model = YOLO(PT_MODEL_PATH)
    print("[INFO] 模型加载完成")
    print()

    # 导出为 ONNX 格式
    print("[INFO] 开始导出 ONNX...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        opset=12,
        simplify=True,
    )
    print(f"[INFO] ONNX 导出完成: {onnx_path}")
    print()

    # 将 ONNX 文件复制到 quant_convert 目录
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    dest_path = os.path.join(ONNX_OUTPUT_DIR, ONNX_OUTPUT_NAME)
    shutil.copy2(onnx_path, dest_path)
    print(f"[INFO] 已复制 ONNX 文件到: {dest_path}")

    # 打印文件大小
    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[INFO] ONNX 文件大小: {file_size_mb:.2f} MB")
    print()
    print("=" * 60)
    print("转换完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()