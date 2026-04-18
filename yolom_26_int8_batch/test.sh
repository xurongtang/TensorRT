./build/test_yolo26_int8_batch \
    ./quant_convert/yolom_int8_batch_4.engine \
    ./asset/000000002261.jpg \
    ./asset/000000018519.jpg \
    ./asset/000000011149.jpg \
    ./asset/000000085772.jpg


# === YOLO26 INT8 Batch Inference Test ===
# Engine: ./quant_convert/yolom_int8_batch_4.engine
# Batch size: 4
# Successfully loaded engine: ./quant_convert/yolom_int8_batch_4.engine
# YOLOM26Int8Batch initialized. batch=4 input=images output=output0
# Loaded image 1: ./asset/000000002261.jpg (640x427)
# Loaded image 2: ./asset/000000018519.jpg (515x640)
# Loaded image 3: ./asset/000000011149.jpg (500x375)
# Loaded image 4: ./asset/000000085772.jpg (640x427)

# --- Single Image Inference ---
# Single inference: 27.7743 ms, 1 detections
# Saved: result_int8_single.jpg

# --- Batch Inference (4 images) ---
# Batch inference: 29.6612 ms total, 7.41529 ms per image
# Image 0: 1 detections -> result_int8_batch_0.jpg
# Image 1: 1 detections -> result_int8_batch_1.jpg
# Image 2: 5 detections -> result_int8_batch_2.jpg
# Image 3: 4 detections -> result_int8_batch_3.jpg

# Done!