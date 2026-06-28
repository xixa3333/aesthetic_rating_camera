import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from core.model import get_model

def export_to_onnx():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    weight_path = os.path.join(os.path.dirname(__file__), '..', config['paths']['best_model'])
    target_size = 512

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx')
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{model_name}_{target_size}.onnx")

    if not os.path.exists(weight_path):
        print(f"找不到權重檔案: {weight_path}")
        return

    device = torch.device("cpu")
    mobile_model = get_model(model_name=model_name, pretrained=False)
    mobile_model.load_state_dict(torch.load(weight_path, map_location=device))
    mobile_model.eval()

    dummy_input = torch.randn(1, 3, target_size, target_size, dtype=torch.float32)

    print(f"開始匯出動態 Batch ONNX 至 {onnx_path}...")
    torch.onnx.export(
        mobile_model, dummy_input, onnx_path,
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX 匯出成功！")

if __name__ == "__main__":
    export_to_onnx()