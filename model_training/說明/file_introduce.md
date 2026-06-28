# 專案檔案結構與目錄說明

本文件提供本深度學習專案的詳細檔案與目錄結構說明。專案主要涵蓋基於 MobileNet 架構的模型訓練、推論、資料處理（AVA 資料集）以及支援多平台部署的模型格式轉換（ONNX, TFLite）。

## 目錄結構概覽


```

```text
File generated successfully.

```text
模型code/
├── checkpoints/         # 模型訓練權重存檔
├── core/                # 專案核心程式模組
├── tools/               # 推論與前處理與資料探勘的腳本
├── data/                # 資料集存放與相關說明
├── export_tools/        # 模型格式轉換與匯出工具
├── models/              # 匯出後的部署模型 (ONNX, TFLite)
├── plot/                # 訓練過程的數據視覺化圖表
├── 說明/                # 說明文件
├── config.yaml          # 全域設定檔
├── main.py              # 主程式進入點
├── requirements.txt     # Python 套件依賴清單
├── test.jpg             # 測試用影像範例

```

---

## 詳細檔案與目錄解釋

### 1. 根目錄 (`模型code/`)

專案的最外層目錄，包含執行訓練、推論的主程式以及環境設定檔。

* **`main.py`**: 專案的執行進入點，負責協調資料載入、模型建構與啟動訓練流程。
* **`config.yaml`**: 集中管理專案的超參數與設定檔，包含學習率、批次大小 (Batch Size)、資料路徑等配置。
* **`requirements.txt`**: 列出執行此專案所需的 Python 函式庫與特定版本，供環境建置使用。
* **`test.jpg`**: 用於快速驗證 `inference.py` 推論功能的測試用輸入影像。

### 2. 核心腳本 (`模型code/tools/`)

* **`inference.py`**: 模型推論腳本，用於載入已訓練的權重，並對新輸入的影像（如 `test.jpg`）進行預測與結果輸出。
* **`prepare_data.py`**: 資料準備腳本，負責執行資料集的前處理、切割與格式轉換。
* **`data_exploration1.py`**: 分數分佈核密度估計（KDE）與視覺化工具。負責將經過 V 型權重重塑後的實際分數分佈，與理論常態分佈進行視覺化對比，用以驗證資料集拉伸後的數學平滑度。
* **`data_exploration2.py`**: 貝氏誤差率（Bayes Error Rate）與底噪精算腳本。透過計算全人類投票的「平均絕對離差（MAD）」，數學量化出資料集不可縮減的物理主觀誤差，為模型 MAE 預測極限提供嚴謹的統計學鐵證。

### 3. 核心模組 (`模型code/core/`)

存放模型訓練與資料處理的核心邏輯程式碼。

* **`dataset.py`**: 定義資料集類別 (Dataset)，負責讀取影像與對應標籤，並提供給 DataLoader 進行批次處理。
* **`model.py`**: 定義神經網路結構（主要為 MobileNet 的各項變體與自定義層）。
* **`trainer.py`**: 封裝訓練迴圈 (Training Loop)、驗證邏輯、損失函數計算與反向傳播過程。
* **`plotter.py`**: 數據視覺化模組，負責在訓練過程中或結束後繪製 Loss 與 Accuracy 等指標曲線。
* **`transforms.py`**: 定義影像的前處理與資料增強 (Data Augmentation) 策略。
* **`__pycache__/`**: Python 編譯後的快取檔案目錄，用於加速模組載入（可忽略）。

### 4. 模型檢查點 (`模型code/checkpoints/`)

* **`best_mobilenet_512.pth`**: 訓練過程中依據驗證集表現所儲存的最佳模型權重檔 (PyTorch 格式)，可用於後續的推論或模型轉換。

### 5. 資料集目錄 (`模型code/data/`)

* **`AVA資料集連結.txt`**: 提供 AVA (Aesthetic Visual Analysis) 影像美學資料集的下載網址或存取方式說明。
* **`目錄結構.png`**: 專案或資料集結構的圖形化說明檔。

### 6. 圖表輸出 (`模型code/plot/`)

存放由 `plotter.py` 生成的訓練過程紀錄圖表，用於分析模型的收斂狀態與效能。

* 包含多個版本的損失函數收斂曲線圖（如 `v01_loss_curve_mobilenet.png` 到 `v13_loss_curve_mobilenet_total.png`），涵蓋了不同訓練策略、損失函數（如 Huber Loss）配置下的訓練軌跡。

### 7. 模型轉換工具 (`模型code/export_tools/`)

提供將訓練完成的 PyTorch 模型轉換為其他框架部署格式的實用工具。

* **`export_onnx.py`**: 負責將 `.pth` 模型轉換為 ONNX 格式，以利跨平台部署。
* **`bypass_and_convert.py`**: 處理模型轉換過程中的特定操作、圖層替換或優化邏輯。
* **`test_tflite.py`**: 用於驗證轉換後的 TFLite 模型推論結果是否與原始模型保持一致。

### 8. 部署模型儲存區 (`模型code/models/`)

存放已轉換完畢、可直接用於端側設備或伺服器部署的模型檔案。

* **`onnx/`**: 包含轉換後的 ONNX 模型檔（如 `mobilenet_512.onnx` 及其簡化版 `mobilenet_512_simp.onnx`）。
* **`tflite/`**: 包含 TensorFlow Lite 格式的模型檔（支援 `float16` 與 `float32` 量化版本的 `mobilenet_512_simp_float*.tflite`），以及 TensorFlow SavedModel 格式的原始檔案 (`.pb` 與 `variables/`)，適合部署於邊緣裝置或行動設備。