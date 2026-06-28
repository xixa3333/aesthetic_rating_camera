import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from core.dataset import AVADataset, DatasetWrapper, get_ava_transforms
from core.model import get_model, get_loss_function
from core.trainer import Trainer
from core.plotter import plot_losses
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 設定隨機種子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 取得資料
    with open("config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 固定種子
    set_seed(config['train']['seed'])

    # 設定: True 訓練+驗證, False 只測試
    DO_TRAIN = True             
    MODEL_NAME = config['model']['name']

    # --- 設定 ---
    BASE_DATA_DIR = config['dataset']['processed_dir']
    BATCH_SIZE = config['train']['batch_size']
    WORKERS = config['train']['num_workers']

    LOADER_ARGS = {
        'batch_size': BATCH_SIZE,
        'num_workers': WORKERS if os.name != 'nt' else 0,
        'pin_memory': True,
    }
    
    # 自動設定路徑
    os.makedirs("checkpoints", exist_ok=True)
    save_path = config['paths']['best_model']  # 例如 "checkpoints/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---建立資料集---
    train_ds_raw = AVADataset(
        csv_file=os.path.join(BASE_DATA_DIR, 'train.csv'), 
        img_dir=os.path.join(BASE_DATA_DIR, 'train'), 
        config=config,
        transform=None
    )
    val_ds_raw = AVADataset(
        csv_file=os.path.join(BASE_DATA_DIR, 'val.csv'), 
        img_dir=os.path.join(BASE_DATA_DIR, 'val'), 
        config=config,
        transform=None
    )
    test_ds_raw = AVADataset(
        csv_file=os.path.join(BASE_DATA_DIR, 'test.csv'), 
        img_dir=os.path.join(BASE_DATA_DIR, 'test'), 
        config=config,
        transform=None
    )
    
    train_ds = DatasetWrapper(train_ds_raw, transform=get_ava_transforms(config, mode='train'))
    val_ds = DatasetWrapper(val_ds_raw, transform=get_ava_transforms(config, mode='val'))
    test_ds = DatasetWrapper(test_ds_raw, transform=get_ava_transforms(config, mode='val'))
    
    train_loader = DataLoader(train_ds, shuffle=True, **LOADER_ARGS)
    val_loader = DataLoader(val_ds, shuffle=False, **LOADER_ARGS)
    test_loader = DataLoader(test_ds, shuffle=False, **LOADER_ARGS)
    
    # ---建立模型---
    EPOCHS = config['train']['epochs']
    LR = float(config['train']['lr'])
    WD = float(config['train']['weight_decay'])

    model = get_model(model_name=MODEL_NAME, pretrained=True).to(device)
    criterion = get_loss_function(float(config['train']['alpha'])).to(device)
    
    optim_params = model.get_optimizer_params(lr=LR, weight_decay=WD)
    optimizer = torch.optim.AdamW(optim_params)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,        # 每 5 個 Epoch 重置一次學習率
        T_mult=2,      # 下一次週期變成 10、20...
        eta_min=1e-6   # 學習率下限
    )

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, 
        scheduler=scheduler, patience=config['train']['early_patience'],
        config=config
    )

    # ---模型訓練---
    if DO_TRAIN:
        print(f"開始訓練模型: {MODEL_NAME}")
        trainer.fit(epochs=EPOCHS, save_path=save_path)
        full_plot_path = os.path.join(config['paths']['plot_dir'], f"loss_curve_{MODEL_NAME}.png")
        plot_losses(trainer.history, save_path=full_plot_path)
    else:
        trainer.load_weights(save_path)

    # ---測試集與baseline輸出---

    m = trainer.test(test_loader)
    
    print("\n" + "="*60)
    print(f"{'指標':<12} | {'加權平均 (Weighted)':<20}")
    print("-" * 60)
    for k in ['mae', 'rmse', 'srcc', 'lcc']:
        print(f"{k.upper():<12} | {m[f'w_{k}']:<20.4f}")
            
    all_test_labels = np.array(m['w_labels'])
    all_random_preds = np.random.permutation(all_test_labels)

    from scipy.stats import spearmanr, pearsonr
    baseline_mae = np.mean(np.abs(all_random_preds - all_test_labels))
    baseline_rmse = np.sqrt(np.mean((all_random_preds - all_test_labels)**2))
    baseline_srcc, _ = spearmanr(all_random_preds, all_test_labels)
    baseline_lcc, _ = pearsonr(all_random_preds, all_test_labels)

    print(f"\n隨機猜測基準 (Baseline):")
    print(f"MAE : {baseline_mae:.4f} | RMSE: {baseline_rmse:.4f}")
    print(f"SRCC: {baseline_srcc:.4f} | LCC : {baseline_lcc:.4f}")

if __name__ == "__main__":
    main()