import torch
from tqdm import tqdm
import os
from scipy.stats import spearmanr, pearsonr
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, patience=3, scheduler=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.patience = patience
        self.counter = 0
        self.best_val_srcc = -1.0

        self.history = {'train_loss':[], 'val_loss':[], 'train_huber':[], 'val_huber':[], 'train_lcc':[], 'val_lcc':[]}
        
        self.train_losses = []
        self.val_losses = []

    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"已成功載入模型權重: {path}")
        else:
            print(f"找不到權重檔案: {path}")
            raise FileNotFoundError

    def train_epoch(self):
        self.model.train()
        running_loss, running_huber, running_lcc = 0.0, 0.0, 0.0
        pbar = tqdm(self.train_loader, desc="Training")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            
            loss, h_loss, l_loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            running_huber += h_loss.item()
            running_lcc += l_loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        n = len(self.train_loader)
        return running_loss / n, running_huber / n, running_lcc / n
    
    def validate(self):
        self.model.eval()
        val_loss, val_huber, val_lcc = 0.0, 0.0, 0.0
        total_mae, total_mse, count = 0.0, 0.0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validating"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                p_raw = self.model(imgs)
                
                loss, h_loss, l_loss = self.criterion(p_raw, labels)
                val_loss += loss.item()
                val_huber += h_loss.item()
                val_lcc += l_loss.item()

                total_mae += torch.abs(p_raw - labels).sum().item()
                total_mse += torch.pow(p_raw - labels, 2).sum().item()
                count += labels.size(0)

                all_preds.extend(p_raw.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        srcc, _ = spearmanr(all_preds, all_labels)
        lcc, _ = pearsonr(all_preds, all_labels)
        
        n = len(self.val_loader)
        return {
            'loss': val_loss / n, 'huber': val_huber / n, 'lcc': val_lcc / n,
            'mae': total_mae / count, 'rmse': np.sqrt(total_mse / count),
            'srcc': srcc, 'lcc_metric': lcc
        }

    # 跑訓練測試迴圈
    def fit(self, epochs, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            t_loss, t_huber, t_lcc = self.train_epoch()
            v_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            # 記錄所有 Loss
            self.history['train_loss'].append(t_loss)
            self.history['train_huber'].append(t_huber)
            self.history['train_lcc'].append(t_lcc)
            self.history['val_loss'].append(v_metrics['loss'])
            self.history['val_huber'].append(v_metrics['huber'])
            self.history['val_lcc'].append(v_metrics['lcc'])

            print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_metrics['loss']:.4f}")
            print(f"Val SRCC: {v_metrics['srcc']:.4f} | Val MAE: {v_metrics['mae']:.4f}")

            # 早停指標為 SRCC
            if v_metrics['srcc'] > self.best_val_srcc:
                self.best_val_srcc = v_metrics['srcc']
                torch.save(self.model.state_dict(), save_path)
                self.counter = 0 
                print(f"模型進步！最佳 SRCC 刷新為 {self.best_val_srcc:.4f}，已儲存至: {save_path}")
            else:
                self.counter += 1
                print(f"SRCC 未上升 ({self.counter}/{self.patience})")
                if self.counter >= self.patience:
                    print("觸發早停，訓練結束。")
                    break

    def test(self, test_loader):
        """ 測試階段：連續數值評判指標 (MAE, RMSE, SRCC, LCC)"""
        self.model.eval()
        total_loss, count = 0.0, 0
        res = {'preds': [], 'labels': []}
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                p_raw = self.model(inputs)  
                
                loss, _, _ = self.criterion(p_raw, labels)
                total_loss += loss.item()  

                res['preds'].extend(p_raw.cpu().numpy())
                res['labels'].extend(labels.cpu().numpy())
                count += labels.size(0)

        # --- 計算指標 ---
        metrics = {'loss': total_loss / len(test_loader)}
        
        p, l = np.array(res['preds']), np.array(res['labels'])
        metrics['w_mae'] = np.mean(np.abs(p - l))
        metrics['w_rmse'] = np.sqrt(np.mean((p - l)**2))
        metrics['w_srcc'], _ = spearmanr(p, l)
        metrics['w_lcc'], _ = pearsonr(p, l)
        metrics['w_labels'] = res['labels'] 
        
        return metrics