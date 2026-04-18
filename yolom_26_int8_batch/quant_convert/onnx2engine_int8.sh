# 先在本地build
rm -f calib.table
./build/onnx2engine_int8 ./yolo26m.onnx ./yolom_int8_batch_4.engine ./dataset 4 calib.table