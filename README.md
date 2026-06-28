# 影像美學評估與智慧終端即時構圖評分系統 (Aesthetic Evaluation & Smart Camera AI)

> **讓每一次快門，都精準捕捉美學極致** —— 本專案是一個結合深度學習**影像美學評估（Aesthetic Quality Assessment, AQA）**與 **Flutter 行動端邊緣運算**的完整端到端系統。在使用者移動鏡頭運鏡的過程中，系統能提供即時、低延遲的浮動美學構圖分數，協助拍攝者在完美構圖誕生的瞬間按下快門。

---

## 項目亮點與核心技術

1. **擺脫保守給分盲區**：透過**分佈重塑（V型權重/Min-Max拉伸）**、分層學習率與相對排序損失函數，訓練出審美觀與人類高度一致、具備大膽給分能力的輕量化推論模型。
2. **高保真影像適配**：採用 `MeanPadToSquare` 演算法進行影像空間自適應補齊，使用影像邊緣平均色填充為正方形，避免傳統強行縮放導致的構圖比例扭曲。
3. **邊緣端即時低延遲推論**：將 PyTorch 訓練出的最佳模型轉換為輕量化 ONNX 與 TensorFlow Lite 格式，在 Android 實體機上實現免連網、高流暢度的即時美學評分。
4. **高並發非同步架構**：Flutter App 端採用微型 MVC 架構，利用 Dart Isolate 獨立線程進行高並發運算，確保相機預覽流與 AI 推論數據管線互不干擾、零卡頓。

---

## 專案架構與目錄說明

本專案主要由兩個核心部分組成：`model_training`（AI 模型訓練與導出）與 `app`（Flutter 相機應用程式）。

```text
aesthetic_rating_camera/
├── model_training/       # 深度學習模型訓練端 (PyTorch)
│   ├── core/             # 資料處理與訓練核心模組 (dataset, model, trainer, transforms等)
│   ├── tools/            # 推論驗證與資料探勘腳本 (inference, data_exploration等)
│   ├── export_tools/     # 模型格式轉換工具 (Export ONNX, TFLite)
│   ├── checkpoints/      # 模型訓練權重存檔
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
## GitHub Releases 釋出產物說明
在專案的 [GitHub Releases] 頁面中，我們提供了已訓練編譯完成的各階段核心權重、跨平台部署模型以及直接可安裝的 Android 應用程式：

## 深度學習權重與模型 (Models)
best_mobilenet_512.pth

說明：基於 PyTorch 框架，在 AVA (Aesthetic Visual Analysis) 資料集上訓練出的核心最佳權重存檔（輸入解析度 512x512）。

mobilenet_512.onnx

說明：自 PyTorch 轉換而來的標準 ONNX 格式模型，適合桌面端、伺服器端或進行跨平台通用推論驗證。

mobilenet_512_simp.onnx

說明：經過 onnx-simplifier 優化後的精簡版 ONNX 模型，消除了冗餘算子、融合了部分層，結構更清晰且執行效率更高。

float32.tflite

說明：標準 FP32 全精度 TensorFlow Lite 模型，用於在行動端/邊緣裝置上進行高精度的美學推論。

float16.tflite

說明：經過 FP16 半精度量化（Quantization）的 TensorFlow Lite 模型。在極小幅犧牲精度的情況下，體積縮減近 50%，並能大幅活化行動端 GPU/NPU 的硬體加速潛能，提供極速即時預覽評分。

## 行動端應用程式 (Application)
SmartCamera_v1.0_Release.apk

說明：已簽章、可直接發佈的 Android 安裝包（建置版本：v1.0.0-Release）。支援 Android 10 到 14+（API Level 29 ~ 34）。內置 float16.tflite 推論大腦，可直接下載安裝於實體手機上進行即時構圖與美學打分測試。

## 環境建置與運行指南
1. 模型訓練端 (model_training)
若您想重新訓練模型或進行格式轉換：

環境安裝：

```Bash
cd model_training
pip install -r requirements.txt
```
啟動訓練：
請先在 config.yaml 中配置好 AVA 資料集路徑、安裝好AVA與使用腳本做好前處理，隨後執行：

```Bash
python main.py
```
模型推論測試：
```Bash
python tools/inference.py --image test.jpg
```

匯出模型 (ONNX & TFLite)：
```Bash
python export_tools/export_onnx.py
python export_tools/bypass_and_convert.py
```

2. Flutter 行動相機端 (app)
若您想微調 App 介面或重新編譯 APK：

配置部署模型：
將編譯好的 float16.tflite 放入 App 的資產目錄中，並確認 pubspec.yaml 中已正確聲明資產路徑。

環境依賴安裝：

```Bash
cd app
flutter pub get
```

以 Debug 模式運行（需連接實體 Android 機並開啟 USB 偵錯）：
```Bash
flutter run
```

建置 Release APK：
```Bash
flutter build apk --release
```
編譯完成後，可在 build/app/outputs/flutter-apk/app-release.apk 取得安裝包。

## 核心演算法與技術細節
### 分佈重塑與統計探勘
V 型權重重塑：針對 AVA 資料集中人類投票多呈現常態分佈導致評分「趨中」的盲區，本系統引入非線性重塑機制，擴大極端美與極端醜的分數差異。

統計學極限驗證：專案中的 data_exploration2.py 透過計算全人類投票的平均絕對離差 (MAD)，數學量化出不可縮減的主觀物理底噪（貝氏誤差率），為模型評估指標提供了科學依據。

### 行動端高性能推論架嘗
解耦微型 MVC：App 捨棄複雜的大型狀態管理，將職責清晰劃分為 main.dart（入口）、camera_screen.dart（UI/鏡頭預覽渲染）與 ai_worker.dart（推論引擎）。

高並發 Isolate 數據管線：相機流每秒產生數十幀高解析度影像，若直接在 UI 線程解碼與推論會引發嚴重卡頓。本系統透過建立獨立的 Isolate 線程，並採用純 Dart 實作的 ITU-R BT.601 標準矩陣解碼演算法進行高效色彩轉換，將輕量化模型推論完全隔絕在背景執行，達成極致流暢的拍攝體驗。

## 團隊與版權資訊
交付源碼版本：v1.0.0-Release