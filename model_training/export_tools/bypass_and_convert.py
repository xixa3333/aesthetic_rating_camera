import os
import onnx2tf
import onnxsim
import onnx
import yaml

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
target_size = 512

onnx_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx')
tflite_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'tflite')
os.makedirs(tflite_dir, exist_ok=True)

raw_onnx_path = os.path.join(onnx_dir, f"{model_name}_{target_size}.onnx")
simp_onnx_path = os.path.join(onnx_dir, f"{model_name}_{target_size}_simp.onnx")

print("1. 正在手動優化 ONNX 模型...")
model = onnx.load(raw_onnx_path)
model_simp, check = onnxsim.simplify(model)
if check:
    onnx.save(model_simp, simp_onnx_path)
    print("ONNX 優化成功！")
else:
    print("ONNX 優化失敗！")

print("2. 正在確保 onnx2tf 測試函數已閹割...")
pkg_dir = os.path.dirname(onnx2tf.__file__)
target_file = os.path.join(pkg_dir, "utils", "common_functions.py")
with open(target_file, 'r', encoding='utf-8') as f:
    code = f.read()

old_func = "def download_test_image_data() -> np.ndarray:"
new_func = """def download_test_image_data() -> np.ndarray:\n    import numpy as np\n    return np.random.rand(1, 3, 224, 224).astype(np.float32)\ndef _disabled_download() -> np.ndarray:"""

if old_func in code and "np.random.rand" not in code:
    code = code.replace(old_func, new_func)
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(code)
    print("源碼閹割成功！")

print("\n請在專案根目錄執行以下終極轉換指令：")
print(f"onnx2tf -i {simp_onnx_path} -o {tflite_dir} -cotof")