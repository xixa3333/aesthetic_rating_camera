import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import ImageFile
import warnings
import yaml
import torchvision.io as io

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

class DatasetWrapper(Dataset):
    """
    用於在 random_split 後，為不同的 Subset 套用不同的 Transform。
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
class AVADataset(Dataset):
    """
    AVA 美學評估資料集：具備資料自動對齊與即時標籤計算功能。
    """
    def __init__(self, csv_file, img_dir, config, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.bin_weights = get_importance_weights()
        self.config = config  
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        raw_id = self.data_frame.iloc[idx]['image_num']
        img_path = os.path.join(self.img_dir, f"{int(raw_id)}.jpg")

        image = io.read_image(img_path, mode=io.ImageReadMode.RGB)

        # 1. 取得 1 到 10 分的原始票數
        votes = self.data_frame.iloc[idx][
            ['vote_1', 'vote_2', 'vote_3', 'vote_4', 'vote_5', 
             'vote_6', 'vote_7', 'vote_8', 'vote_9', 'vote_10']
        ].values.astype(float)
        
        # 2. 轉為張量並套用權重重塑
        probs = torch.tensor(votes, dtype=torch.float32)
        
        label_dist = reshape_distribution(probs, self.bin_weights)

        score_range = torch.arange(1, 11, dtype=torch.float32)
        raw_mean_score = torch.sum(label_dist * score_range).item()

        scaled_mean_score = scale_to_aesthetic_range(raw_mean_score, self.config)

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(scaled_mean_score, dtype=torch.float32)

def get_ava_transforms(config, mode='train'):
    base_transform = [
        transforms.ConvertImageDtype(torch.float32)
    ]
    
    # 有做資料擴增
    if mode == 'train':
        base_transform.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0, hue=0),
        ])
    
    base_transform.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(base_transform)

def get_importance_weights():
    """
    獲取用於平衡常態分佈的逆頻率權重。
    此配置旨在將 AVA 的鐘形分佈拉平，強化兩極影響力。
    """
    #22.50, 10.63, 4.34, 1.74, 1.00, 1.27, 2.47, 5.09, 12.08, 20.13
    #1,1,1,1,1,1,1,1,1,1
    return torch.tensor([22.50, 10.63, 4.34, 1.74, 1.00, 1.27, 2.47, 5.09, 12.08, 20.13], dtype=torch.float32)

def reshape_distribution(probs, weights):
    """
    對機率分佈執行重要性重塑並重新歸一化。
    """
    weighted_probs = probs * weights
    return weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)

def scale_to_aesthetic_range(raw_score, config):
    """
    將模型或標籤計算出的原始趨中分數，
    線性拉伸到標準的 1-10 分美學宇宙。
    
    支援傳入：單一數值 (float) 或 NumPy 陣列 / PyTorch 張量。
    """
    min_score = float(config['dataset']['min_score'])
    max_score = float(config['dataset']['max_score'])
    
    # Min-Max 映射至 1-10 區間
    scaled = (raw_score - min_score) * 9.0 / (max_score - min_score) + 1.0
    
    if isinstance(raw_score, torch.Tensor):
        return torch.clamp(scaled, 1.0, 10.0)
    elif isinstance(raw_score, np.ndarray):
        return np.clip(scaled, 1.0, 10.0)
    return max(1.0, min(10.0, scaled))

# --- 測試區 (執行 python core/dataset.py 即可驗證) ---
if __name__ == "__main__":
    csv_path = 'data/ground_truth_dataset.csv'
    img_dir = 'data/images'
    
    if os.path.exists(csv_path) and os.path.exists(img_dir):
        config = yaml.safe_load(open("config.yaml"))
        ds = AVADataset(csv_path, img_dir, config)
        img, label = ds[0]
        print(f"測試成功！張量維度: {img.shape}, 標籤分數: {label.item():.4f}")
    else:
        print("請確認路徑是否存在。")