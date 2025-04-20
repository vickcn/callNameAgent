# 繁體字識別系統

這是一個基於OpenCV和Tesseract OCR的繁體字識別系統，可以實時識別攝像頭中的繁體字。

## 功能特點

- 實時繁體字識別
- 特徵分析（筆畫密度、結構複雜度、方向梯度等）
- 語音播報識別結果
- 圖像預處理優化

## 技術棧

- 後端：Python, Flask, OpenCV, Tesseract OCR
- 前端：React, TypeScript
- 實時通信：Socket.IO

## 安裝說明

1. 安裝Python依賴：
```bash
pip install -r requirements.txt
```

2. 安裝Tesseract OCR：
```bash
# macOS
brew install tesseract
brew install tesseract-lang  # 安裝語言包

# Ubuntu
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-tra  # 安裝繁體中文語言包
```

3. 安裝前端依賴：
```bash
cd frontend
npm install
```

## 運行說明

1. 啟動後端服務：
```bash
python backend/app.py
```

2. 啟動前端服務：
```bash
cd frontend
npm start
```

3. 在瀏覽器中訪問：`http://localhost:3000`

## 項目結構

```
.
├── backend/            # 後端代碼
│   ├── app.py         # 主應用程序
│   └── requirements.txt
├── frontend/          # 前端代碼
│   ├── src/          # 源代碼
│   └── package.json
└── README.md
```

## 貢獻指南

1. Fork 本項目
2. 創建新的分支
3. 提交更改
4. 發起 Pull Request

## 許可證

MIT License
