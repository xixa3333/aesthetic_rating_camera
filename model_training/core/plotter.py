import matplotlib.pyplot as plt
import os

def plot_losses(history, save_path="loss_curve.png"):
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    name, ext = os.path.splitext(save_path)
    path_total = f"{name}_total{ext}"
    path_huber = f"{name}_huber{ext}"
    path_lcc = f"{name}_lcc{ext}"

    # 1. Total Loss 圖
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['val_loss'], label='Val Total Loss')
    plt.title('Training and Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss (Hybrid)')
    plt.legend()
    plt.grid(True)
    plt.savefig(path_total)
    plt.close()

    # 2. Huber Loss 圖
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_huber'], label='Train Huber Loss')
    plt.plot(history['val_huber'], label='Val Huber Loss')
    plt.title('Training and Validation Huber Loss (Absolute Error)')
    plt.xlabel('Epochs')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path_huber)
    plt.close()

    # 3. LCC Loss 圖
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_lcc'], label='Train LCC Loss')
    plt.plot(history['val_lcc'], label='Val LCC Loss')
    plt.title('Training and Validation LCC Loss (Rank Correlation)')
    plt.xlabel('Epochs')
    plt.ylabel('LCC Loss (1 - Pearson)')
    plt.legend()
    plt.grid(True)
    plt.savefig(path_lcc)
    plt.close()

    print(f"已輸出三張損失曲線圖至: {dir_name}")