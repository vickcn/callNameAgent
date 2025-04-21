# 繁體字辨識系統

這是一個基於OpenCV和Tesseract OCR的繁體字辨識系統，可以即時辨識攝影機中的繁體字。

## 功能特點

- 即時繁體字辨識
- 特徵分析（筆畫密度、結構複雜度、方向梯度等）
- 語音播報辨識結果
- 圖像預處理最佳化

## 技術棧

- 後端：Python, Flask, OpenCV, Tesseract OCR
- 前端：React, TypeScript
- 即時通訊：Socket.IO

## 安裝說明

1. 安裝Python相依套件：
```bash
pip install -r requirements.txt
```

2. 安裝Tesseract OCR：
```bash
# macOS
brew install tesseract
brew install tesseract-lang

# Ubuntu
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-tra
```

3. 安裝FFmpeg（用於影像處理）：
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get update
sudo apt-get install ffmpeg

# Windows（使用Chocolatey）
choco install ffmpeg
```

4. 安裝前端相依套件：
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

## 專案結構

```
.
├── backend/            # 後端程式碼
│   ├── app.py         # 主應用程式
│   └── requirements.txt
├── frontend/          # 前端程式碼
│   ├── src/          # 原始碼
│   └── package.json
└── README.md
```

## 貢獻指南

1. Fork 本專案
2. 建立新的分支
3. 提交更改
4. 發起 Pull Request

## 授權條款

MIT License 