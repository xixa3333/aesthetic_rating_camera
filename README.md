# 影像美學即時評分系統

> **讓每一次快門，都精準捕捉美學極致** —— 本專案是一個結合深度學習**影像美學評估**與 **Flutter 行動端邊緣運算**的端到端系統。在使用者移動鏡頭運鏡的過程中，系統能提供即時、低延遲的浮動美學構圖分數，協助拍攝者在完美構圖誕生的瞬間按下快門。

---

## 項目亮點與核心技術

1. **擺脫保守給分盲區**：透過**分佈重塑（V型權重/Min-Max拉伸）**與分層學習率優化，迫使神經網路學習更具邊界感與大膽的審美特徵，解決群體審美趨中的平庸性問題。
2. **Hybrid Aesthetic Loss 複合審美損失**：結合自訂 **Huber 損失函數**與**線性相關係數損失 (LCC Loss)**。Huber 篩選機制自動放棄擬合主觀噪點，防止梯度爆炸；LCC 則大幅提升模型對不同圖片之間美學相對排序的敏感度。
3. **輕量化邊緣端骨幹網路**：選用適合行動端部署的 **MobileNetV3** 作為特徵提取網路，並建置 **Multi-Sample Dropout 迴歸層**以抑制模型過擬合。
4. **高保真影像幾何對齊**：採用 **MeanPadToSquare** 演算法進行影像空間自適應補齊，使用影像邊緣平均色填充為正方形，徹底捨棄傳統暴力縮放或隨機裁剪，完美保留相片原始的黃金構圖比例與線條延展性。
5. **邊緣端即時低延遲推論**：將訓練出的最佳模型轉換為輕量化 ONNX 與 TensorFlow Lite 格式，在 Android 實體機上實現免連網、高流暢度的即時美學評分。
6. **高並發非同步架構**：Flutter App 端採用微型 MVC 架構，利用 **Dart Isolate** 獨立線程進行背景非同步運算。採用純 Dart 實作的 **ITU-R BT.601 標準矩陣解碼演算法**進行高效色彩轉換（YUV 轉 RGB），確保相機預覽流與 AI 推論管線互不干擾、零卡頓。

---

## 專案架構與目錄說明

本專案主要由兩個核心部分組成：`model_training`（AI 模型訓練與導出）與 `app`（Flutter 相機應用程式）。

```text
aesthetic_rating_camera/
├── model_training/       # 深度學習模型訓練端 (PyTorch)
│   ├── core/             # 資料處理與訓練核心模組 (dataset, model, trainer, transforms等)
│   ├── tools/            # 推論驗證與資料探勘腳本 (inference, data_exploration等)
│   ├── export_tools/     # 模型格式轉換工具 (Export ONNX, TFLite)
│   ├── checkpoints/      # 模型權重存檔
│   ├── config.yaml       # 全域超參數設定檔
│   ├── requirements.txt  # Python 環境依賴清單
│   └── main.py           # 訓練主程式進入點
│
└── app/                  # 智慧終端行動相機端 (Flutter)
    ├── lib/              # Dart 核心源碼
    │   ├── main.dart           ➔ 系統生命週期入口與全域硬體參數配置
    │   ├── camera_screen.dart  ➔ 即時相機流渲染、動態手勢與 UI 介面
    │   └── ai_worker.dart      ➔ 高並發 Isolate 運算進程與 TFLite 推論管線
    └── pubspec.yaml      # Flutter 套件依賴配置

```

---

## 實驗結果與效能指標

本系統基於大規模影像美學資料集 **AVA** 的 25.5 萬張群體投票數據進行訓練與迭代。

### 核心數據演進

| 指標/參數 | v0 (Baseline) | v1 (初版) | v5 (解析度提升) | v7 (EMD探索) | v8 (標籤拉伸) | v13 (最終優化版) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Y標籤處理** | 無 | 無 | 無 | 無 | 加權前處理 | 加權-拉伸處理 |
| **圖片前處理** | 無 | 模糊補齊 | 純色補齊 | 純色補齊 | 純色補齊 | 純色補齊 (MeanPad) |
| **輸入尺寸** | 無 | 224x224 | 512x512 | 512x512 | 512x512 | 512x512 |
| **損失函數** | 無 | Huber | Huber | EMD | EMD | **Huber + LCC Loss** |
| **MAE** | 1.8578 | 0.4629 | 0.4167 | 0.4139 | 0.8981 | **0.9484** (拉伸區間) |
| **SRCC** | 0.0053 | 0.5857 | 0.6769 | **0.6792** | 0.6459 | 0.6685 |
| **LCC** | 0.0055 | - | - | 0.6930 | 0.6592 | **0.6796** |

### 技術評估與物理意義

1. **打破人類主觀分歧底噪**：資料統計顯示，AVA 資料集的人類群體投票平均絕對離差 (MAD) 高達 2.0 分。本模型最終版本在拉伸後的資料集上達成了 **0.948 分的 MAE**，遠低於人類自身的分歧度。透過 Huber 損失函數的篩選機制，模型成功過濾了個體主觀偏見，精準鎖定影像純視覺美學的客觀大眾共識。


2. **非視覺隱藏變數免疫性**：模型純粹依據構圖、光影、色彩空間張力等視覺客觀特徵進行給分，完全免疫了人類因相片背後的情緒標籤、故事說明所產生的情感偏誤與主題綁架，確立了本系統作為「客觀視覺構圖裁判」的科學定位。



---

## GitHub Releases 釋出產物說明

在專案的 [GitHub Releases] 頁面中，我們提供了編譯完成的核心權重、跨平台部署模型以及 Android 應用程式安裝包：

### 深度學習權重與模型 (Models)

* **`best_mobilenet_512.pth`**：基於 PyTorch 框架，在 AVA 資料集上訓練出的核心最佳權重存檔（輸入解析度 512x512）。


* **`mobilenet_512.onnx`**：自 PyTorch 轉換而來的標準 ONNX 格式模型，適合桌面端、伺服器端推論驗證。


* **`mobilenet_512_simp.onnx`**：經過 `onnx-simplifier` 優化後的精簡版 ONNX 模型，消除冗餘算子，執行效率更高。


* **`float32.tflite`**：標準 FP32 全精度 TensorFlow Lite 模型，用於行動端/邊緣裝置的高精度美學推論。


* **`float16.tflite`**：經過 FP16 半精度靜態量化的 TensorFlow Lite 模型。在極小幅犧牲精度的情況下，**體積縮減近 50%**，大幅活化行動端 GPU/NPU 的硬體加速潛能，提供極速即時預覽評分。



### 行動端應用程式 (Application)

* **`SmartCamera_v1.0_Release.apk`**：已簽章、可直接發布的 Android 安裝包（建置版本：v1.0.0-Release）。支援 Android 10 到 14+（API Level 29 ~ 34）。內置 `float16.tflite` 推論大腦，可直接安裝於實體手機上進行免連網、零卡頓的即時構圖與美學打分測試。

---

## 環境建置與運行指南

### 1. 模型訓練端 (model_training)

若您想重新訓練模型或進行格式轉換：

* **環境安裝**：
```bash
cd model_training
pip install -r requirements.txt

```


* **啟動訓練**：
請先在 `config.yaml` 中配置好 AVA 資料集路徑、並完成資料前處理，隨後執行：


```bash
python main.py

```


* **模型推論測試**：
```bash
python tools/inference.py --image test.jpg

```


* **匯出模型 (ONNX & TFLite)**：
```bash
python export_tools/export_onnx.py
python export_tools/bypass_and_convert.py

```



### 2. Flutter 行動相機端 (app)

若您想微調 App 介面或重新編譯 APK：

* **配置部署模型**：
將編譯好的 `float16.tflite` 放入 App 的資產目錄中，並確認 `pubspec.yaml` 中已正確聲明資產路徑。


* **環境依賴安裝**：
```bash
cd app
flutter pub get

```


* **以 Debug 模式運行**（需連接實體 Android 機並開啟 USB 偵錯）：


```bash
flutter run

```


* **建置 Release APK**：


```bash
flutter build apk --release

```

編譯完成後，可在 `build/app/outputs/flutter-apk/app-release.apk` 取得安裝包。