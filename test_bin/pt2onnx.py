from ultralytics import YOLO

# 加载 YOLOv11 分割模型
model = YOLO("./yolo11s.pt")

# 导出onnx
model.export(format="onnx")
