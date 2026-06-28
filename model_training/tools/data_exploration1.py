import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from matplotlib.widgets import Slider
from scipy.stats import gaussian_kde
from core.dataset import get_importance_weights, reshape_distribution
import yaml

with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

def calculate_kde_weights(data, min_score=1, max_score=9, clip_value=9.0):
    """
    基於數據真實分佈 (KDE) 計算補償權重
    """
    # 1. 訓練 KDE 模型
    kde = gaussian_kde(data)
    
    # 2. 定義評分點 (1-9)
    scores = np.arange(min_score, max_score + 1)
    
    # 3. 計算每個分數點的密度值
    # 注意：如果數據量極少，此處數值會趨近於 0
    densities = kde.evaluate(scores)
    
    # 4. 計算原始權重 (密度的倒數)
    # 加上一個極小值 epsilon 避免除以 0
    raw_weights = 1.0 / (densities + 1e-10)
    
    # 5. 歸一化權重：讓平均權重為 1，保持總體 Loss 規模穩定
    normalized_weights = raw_weights / np.mean(raw_weights)
    
    # 6.限制權重上限
    # 避免邊緣極端樣本導致訓練爆炸
    final_weights = np.clip(normalized_weights, 0, clip_value)
    
    return scores, densities, final_weights

def plot_interactive_distribution(csv_path):
    if not os.path.exists(csv_path):
        print(f"找不到檔案: {csv_path}")
        return

    # 1. 數據處理
    df = pd.read_csv(csv_path)
    vote_cols = [f'vote_{i}' for i in range(1, 11)]
    votes_tensor = torch.tensor(df[vote_cols].values, dtype=torch.float32)
    
    row_sums = votes_tensor.sum(dim=-1, keepdim=True)
    probs = torch.where(row_sums != 0, votes_tensor / row_sums, torch.zeros_like(votes_tensor))
    
    weights = get_importance_weights()
    final_probs = reshape_distribution(probs, weights)
    
    score_range = torch.arange(1, 11, dtype=torch.float32)
    weighted_avg_scores = torch.sum(final_probs * score_range, dim=-1).numpy()
    min_score = config['dataset']['min_score']
    max_score = config['dataset']['max_score']
    weighted_avg_scores = np.clip(list(map(lambda x: (x - min_score) * 9.0 / (max_score - min_score) + 1, weighted_avg_scores)), 1, 10)
    
    print(min(weighted_avg_scores), max(weighted_avg_scores))
    
    df['reshaped_score'] = weighted_avg_scores
    df['reshaped_rounded_score'] = np.clip(np.floor(weighted_avg_scores), 1, 9).astype(int)
    print(df['reshaped_rounded_score'].value_counts().sort_index())
    total_samples = len(df)
    
    # 2. 建立圖表與座標軸
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    # 繪製底層柱狀圖
    sns.countplot(data=df, x='reshaped_rounded_score', color='lightgray', order=range(1, 10), ax=ax, alpha=0.5)
    
    # 3. 準備 X 軸數據 (0-8 對應 分數區間 1-9)
    x_plot = np.linspace(0, 8, 500)
    x_values = x_plot + 1 # 實際的數值

    # --- 曲線 A: 數據實際分布 (KDE) ---
    kde = gaussian_kde(weighted_avg_scores)
    data_curve_y = kde.evaluate(x_values) * total_samples
    ax.plot(x_plot, data_curve_y, color='#1f77b4', lw=3, label='Actual Data Distribution (KDE)')
    ax.fill_between(x_plot, 0, data_curve_y, color='#1f77b4', alpha=0.1)

    # --- 曲線 B: 理論常態分布 ---
    mu_init = weighted_avg_scores.mean()
    sigma_init = weighted_avg_scores.std()

    def normal_pdf_scaled(x_v, mu, sigma):
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_v - mu) / sigma)**2)
        return pdf * total_samples

    line_theoretical, = ax.plot(x_plot, normal_pdf_scaled(x_values, mu_init, sigma_init), 
                                'r--', lw=2, label='Theoretical Normal (Adjustable)')

    ax.set_title("Distribution Comparison: Actual KDE vs. Theoretical Normal", fontsize=14, fontweight='bold')
    ax.set_xlabel("Score (1-9)")
    ax.set_ylabel("Frequency / Density Scale")
    ax.legend()

    ax_mu = plt.axes([0.25, 0.12, 0.5, 0.03])
    ax_sigma = plt.axes([0.25, 0.07, 0.5, 0.03])
    s_mu = Slider(ax_mu, 'Theory Mu', 1.0, 9.0, valinit=mu_init)
    s_sigma = Slider(ax_sigma, 'Theory Sigma', 0.1, 5.0, valinit=sigma_init)

    def update(val):
        mu = s_mu.val
        sigma = s_sigma.val
        line_theoretical.set_ydata(normal_pdf_scaled(x_values, mu, sigma))
        fig.canvas.draw_idle()

    s_mu.on_changed(update)
    s_sigma.on_changed(update)

    plt.show()

if __name__ == "__main__":
    plot_interactive_distribution("data/ground_truth_dataset.csv")