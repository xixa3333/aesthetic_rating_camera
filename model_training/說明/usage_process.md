# 美學評分模型 (Aesthetic Scoring Model)

本專案包含基於 PyTorch 的美學評分模型訓練、資料前處理，以及針對行動端 (Flutter) 部署的 TFLite 量化與匯出管線。

**注意**：本專案的 `requirements.txt` 採用環境快照凍結，**僅保證於 Windows 系統下可 100% 重現建置**。若於 macOS 或 Linux 伺服器建置，可能需要手動排除部分系統綁定的相容性套件。

## 環境建置指南

### 步驟 1：建立虛擬環境

請打開終端機 (Anaconda Prompt 或支援 conda 指令的 PowerShell)，並執行以下指令。
這裡強制指定 Python 3.10，並加上 `--no-default-packages` 來保持初始環境極致輕量：

```bash
conda create -n ava_scorer python=3.10 --no-default-packages -y

```

### 步驟 2：啟動虛擬環境

```bash
conda activate ava_scorer

```

*(啟動成功後，你的命令列最前方會出現 (ava_scorer) 標示)*

### 步驟 3：安裝依賴套件

在確認虛擬環境已經啟動的狀態下，執行以下指令安裝所需套件（包含 PyTorch 訓練核心與 ONNX/TFLite 模型轉換工具）：

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install --upgrade pip
pip install -r requirements.txt

```

---

## 執行流程

環境建置完成後，請務必於**專案根目錄**下，依照以下順序執行專案：

### 1. 資料前處理

* 將原始圖片進行 Padding、尺寸縮放，並切割成 Train/Val/Test 資料集。
* 指令：`python -m tools.prepare_data`

### 2. 模型訓練

* 讀取前處理好的資料，開始訓練並自動儲存最佳模型權重至 `checkpoints/`。
* 需進入 `config.yaml` 確認 `DO_TRAIN = True`。
* 指令：`python main.py`

### 3. 單張影像推論 (電腦端驗證)

* 使用訓練好的 PyTorch 模型權重 (`.pth`)，對全新上傳的圖片進行美學評分。
* 指令：`python -m tools.inference`

### 4. 模型量化與邊緣端匯出 (行動端部署準備)

此階段會將 PyTorch 模型轉換為 Flutter APP 專用的 FP16 半精度 TFLite 模型。

* **Step 4-1: 匯出 ONNX 格式**
將模型靜態化並鎖死輸入尺寸為 512x512。
* 指令：`python export_tools/export_onnx.py`
* 產出：`models/onnx/mobilenet_512.onnx`

* **Step 4-2: 圖形優化與 TFLite 轉換**
精簡 ONNX 計算圖，並透過繞過底層測試機制的腳本進行轉換，同時指定 FP16 量化。
* 指令：`python export_tools/bypass_and_convert.py`
* 產出：`models/tflite/mobilenet_512_float16.tflite`

* **Step 4-3: TFLite 健康度驗證**
在 Python 端模擬 Flutter 的 YUV 轉 RGB 與 Padding 前處理，確保量化模型輸出數值未發生崩潰或溢出。
* 指令：`python -m export_tools.test_tflite`



---

## 離開虛擬環境

當你完成工作，想要退出虛擬環境時，只需在終端機輸入：

```bash
conda deactivate

```