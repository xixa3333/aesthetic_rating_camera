import torch
from PIL import Image
import os
import time
import yaml
from core.model import get_model
from core.transforms import get_inference_transform

class AestheticPredictor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = self.config['model']['target_size']
        self.model_name = self.config['model']['name']
        self.weight_path = self.config['paths']['best_model']

        self.model = get_model(model_name=self.model_name, pretrained=False)

        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"找不到權重檔案: {self.weight_path}")

        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        
        self.transform = get_inference_transform(target_size=self.target_size)

    def predict(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            # 推論處理 padding 並直接輸出 0.0~1.0 的 Tensor
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                raw_score = self.model(input_tensor).item()
                final_score = max(1.0, min(10.0, raw_score))
            return final_score
        except Exception as e:
            print(f"預測錯誤 {img_path}: {e}")
            return None

# --- 快速測試 ---
if __name__ == "__main__":
    start=time.time()
    predictor = AestheticPredictor(config_path="config.yaml")
    score = predictor.predict("test.jpg")
    if score:
        end = time.time()        
        print(f"預測完成，耗時 {end - start:.2f} 秒")
        print(f"該影像之美學評分: {score:.2f}")