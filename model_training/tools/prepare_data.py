import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.transforms import MeanPadToSquare
import random
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 設定 ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CSV_PATH = config['dataset']['raw_csv_path']
SRC_IMG_DIR = config['dataset']['raw_img_dir']
TARGET_SIZE = config['model']['target_size']
OUTPUT_BASE_DIR = config['dataset']['processed_dir']
MAX_WORKERS = config['train']['num_workers']

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def process_and_save(args):
    filename, target_subdir = args
    src_path = os.path.join(SRC_IMG_DIR, filename)
    dest_path = os.path.join(OUTPUT_BASE_DIR, target_subdir, filename)
    
    if os.path.exists(dest_path): 
        return filename
    try:
        img = Image.open(src_path).convert('RGB')
        padder = MeanPadToSquare(target_size=TARGET_SIZE)
        img_processed = padder(img)
        img_processed.save(dest_path, "JPEG", quality=95)
        return filename
    except:
        return None

def main():
    set_seed(config['train']['seed'])
    
    # 1. 讀取與過濾
    df = pd.read_csv(CSV_PATH)
    valid_files = {f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith('.jpg')}
    df['filename'] = df['image_num'].apply(lambda x: f"{int(x)}.jpg")
    df = df[df['filename'].isin(valid_files)].reset_index(drop=True)
    
    # 2. 隨機切分索引
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    train_idx = indices[:int(0.8 * len(df))]
    val_idx = indices[int(0.8 * len(df)):int(0.9 * len(df))]
    test_idx = indices[int(0.9 * len(df)):]
    
    splits = [('train', train_idx), ('val', val_idx), ('test', test_idx)]
    
    for name, idx_list in splits:
        subdir = os.path.join(OUTPUT_BASE_DIR, name)
        os.makedirs(subdir, exist_ok=True)
        
        df.iloc[idx_list].to_csv(os.path.join(OUTPUT_BASE_DIR, f'{name}.csv'), index=False)
        
        print(f"準備處理 {name} 集合: {len(idx_list)} 筆資料...")
        
        # 1. 建立該集合的任務清單
        tasks = [(df.iloc[i]['filename'], name) for i in idx_list]
        effective_workers = min(MAX_WORKERS, 4) if os.name == 'nt' else MAX_WORKERS

        # 2. 執行平行處理，並過濾出成功的檔名
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # 1. 提交所有任務並取得 Future 物件清單
            futures = [executor.submit(process_and_save, task) for task in tasks]
            
            # 2. 搭配 as_completed 與 tqdm，達成真正的即時進度更新
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"處理 {name}"):
                results.append(future.result())
        
        successful_files = set([res for res in results if res is not None])
        
        # 3. 確保 CSV 裡只有 100% 成功處理的圖片
        split_df = df.iloc[idx_list]
        clean_df = split_df[split_df['filename'].isin(successful_files)]
        clean_df.to_csv(os.path.join(OUTPUT_BASE_DIR, f'{name}.csv'), index=False)
        print(f"{name} 集合完成: 成功 {len(clean_df)} / 原始 {len(idx_list)}")

if __name__ == "__main__":
    main()