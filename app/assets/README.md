# 📦 匯出部署模型存放區 (Exported Models Directory)

本資料夾為專案模型推論的預設輸入路徑。

## ⚠️ 為什麼這裡是空的？

為保持 Git 儲存庫的輕量化與克隆（Clone）速度，我們已將所有龐大的二進位模型檔排除在版本控制之外。

## 📥 如何獲取預訓練模型？

請至本專案的 [GitHub Releases] 頁面下載對應的權重檔案，並嚴格按照以下目錄結構放置檔案，否則推論腳本將無法執行：

```text
assets/
├── mobilenet_512_simp_float16.tflite