import numpy as np
import tensorflow as tf
from PIL import Image
import os

def test_tflite_model(tflite_path, image_path):
    if not os.path.exists(tflite_path):
        print(f"找不到模型檔案: {tflite_path}")
        return
        
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    print(f"模型預期輸入維度: {input_shape}")

    # 1. 模擬 Flutter 端的影像前處理
    target_size = 512
    img = Image.open(image_path).convert('RGB')
    
    stat = np.array(img).mean(axis=(0, 1)).astype(int)
    bg = Image.new('RGB', (target_size, target_size), tuple(stat))
    
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    bg.paste(img_resized, (offset_x, offset_y))

    # 2. 像素歸一化到 0.0 ~ 1.0 
    input_data = np.array(bg, dtype=np.float32) / 255.0
    
    # 3. 在外部精準執行 PyTorch 標準的 ImageNet 減均值除方差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_data = (input_data - mean) / std
    
    # 4. 根據 onnx2tf 轉換後的真實 TFLite 輸入維度進行適配
    if input_shape[1] == 3: 
        # 如果是 NCHW [1, 3, 512, 512]
        input_data = np.transpose(input_data, (2, 0, 1))
    
    input_data = np.expand_dims(input_data, axis=0)

    # 5. 執行推論
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    raw_score = interpreter.get_tensor(output_details[0]['index'])[0]
    print(f"[{image_path}] TFLite 模型預測美學分數: {raw_score}")

if __name__ == "__main__":
    test_tflite_model("models/tflite/mobilenet_512_simp_float16.tflite", "test.jpg")