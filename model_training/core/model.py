import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# 可做學習率分層
def _get_split_params(module, lr, weight_decay):
    """ 內部輔助函式：將模組參數區分為 decay 與 no_decay 兩組 """
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
    ]

# MobileNetV3 模型
class MobileNetV3Wrapper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)
        in_features = self.backbone.classifier[3].in_features

        self.dropouts = nn.ModuleList([nn.Dropout(p=0.1 + i*0.05) for i in range(5)])
        self.fc = nn.Linear(in_features, 1)
        self.backbone.classifier[3] = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        out = torch.mean(torch.stack([self.fc(drop(x)) for drop in self.dropouts]), dim=0)
        return out.squeeze(-1)
    
    def get_optimizer_params(self, lr, weight_decay):
        """ 封裝 MobileNetV3 的分層學習率與權重衰減設定 """
        optim_params = []
        # Features 層級給予 0.1 倍學習率
        optim_params.extend(_get_split_params(self.backbone.features, lr * 0.1, weight_decay))
        # Classifier 層級給予標準學習率
        optim_params.extend(_get_split_params(self.backbone.classifier, lr, weight_decay))
        return optim_params

# 自己設計的 Custom CNN
class SimpleCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 輸出: 32 x 112 x 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 輸出: 64 x 56 x 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((7, 7)) # 最終固定輸出為 128 x 7 x 7
        )
        
        # Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x).squeeze(-1)
        return x

    def get_optimizer_params(self, lr, weight_decay):
        """ 封裝 CustomCNN 的分層學習率與權重衰減設定 """
        optim_params = []
        optim_params.extend(_get_split_params(self.features, lr * 0.1, weight_decay))
        optim_params.extend(_get_split_params(self.classifier, lr, weight_decay))
        return optim_params

# 透過字串隨時切換
def get_model(model_name="mobilenet", pretrained=True):
    if model_name == "mobilenet":
        return MobileNetV3Wrapper(pretrained=pretrained)
    elif model_name == "custom":
        return SimpleCustomCNN()
    else:
        raise ValueError(f"不支援的模型名稱: {model_name}")

# ---loss---

class HybridAestheticLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-8):
        """
        alpha: 控制 LCC Loss 影響力的比例，範圍需在 0.0 ~ 1.0 之間
        eps: 防止除以零的極小值防呆機制
        """
        super().__init__()
        self.huber = nn.HuberLoss(delta=2.0)
        self.alpha = alpha
        self.eps = eps

    def forward(self, preds, labels):
        # 1. 計算基礎迴歸 Loss
        huber_loss = self.huber(preds, labels)

        if preds.size(0) < 2:
            return huber_loss, huber_loss, torch.tensor(0.0).to(preds.device)
        
        # 2. 計算 Pearson Correlation Coefficient (LCC)
        vx = preds - torch.mean(preds)
        vy = labels - torch.mean(labels)
        
        cov = torch.sum(vx * vy)
        var_preds = torch.sum(vx ** 2) + self.eps
        var_labels = torch.sum(vy ** 2) + self.eps
        
        lcc = cov / (torch.sqrt(var_preds) * torch.sqrt(var_labels))
        lcc_loss = 1.0 - lcc
        
        # 3. 計算 Total Loss
        total_loss = (1.0 - self.alpha) * huber_loss + self.alpha * lcc_loss
        return total_loss, huber_loss, lcc_loss

def get_loss_function(alpha=0.5):
    return HybridAestheticLoss(alpha=alpha)