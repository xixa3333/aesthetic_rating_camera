import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class MeanPadToSquare:
    """ 純 PIL 影像處理：負責計算平均色並補齊為正方形 """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        stat = np.array(img).mean(axis=(0, 1)).astype(int)
        mean_color = tuple(stat)
        bg = Image.new('RGB', (self.target_size, self.target_size), mean_color)
        
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        offset_x = (self.target_size - new_w) // 2
        offset_y = (self.target_size - new_h) // 2
        bg.paste(img_resized, (offset_x, offset_y))
        
        return bg

def get_inference_transform(target_size):
    """ 推論專屬：組合 Padding、轉張量與正規化 """
    return transforms.Compose([
        MeanPadToSquare(target_size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])