import os
import cv2
import numpy as np
import pytesseract
import pyttsx3
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import threading
import re
import logging
import time
from queue import Queue, Empty
import json
from ian_toolkit import LOGger

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 語音任務隊列
speech_queue = Queue()
speech_lock = threading.Lock()
speech_thread = None
# temp_engine = pyttsx3.init()
# temp_engine.setProperty('rate', 150)
# temp_engine.setProperty('volume', 0.9)
def speech_worker():
    """語音工作線程"""
    logger.info("語音工作線程已啟動")
    while True:
        try:
            # 從隊列中獲取文本，設置超時時間
            try:
                text = speech_queue.get(timeout=1)  # 1秒超時
                if text is None:  # 結束信號
                    break
            except Empty:
                continue  # 如果超時，繼續等待
                
            logger.info(f"開始朗讀文本: {text}")
            logger.info(f"當前語音隊列大小: {speech_queue.qsize()}")
            
            try:
                # 每次使用新的引擎實例
                temp_engine = pyttsx3.init()
                temp_engine.setProperty('rate', 150)
                temp_engine.setProperty('volume', 0.9)
                temp_engine.say(text.strip())
                temp_engine.runAndWait()
                temp_engine.stop()
                # del temp_engine
                temp_engine = None
                logger.info(f"完成朗讀文本: {text}")
            except Exception as e:
                logger.error(f"朗讀文本時發生錯誤: {str(e)}")
                # 如果發生錯誤，等待一段時間再繼續
                time.sleep(0.5)
            
            # 短暫延遲，避免連續朗讀太快
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"語音工作線程錯誤: {str(e)}")
            time.sleep(1)  # 發生錯誤時等待一段時間再繼續

def start_speech_thread():
    """啟動語音工作線程"""
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speech_worker, daemon=True)
        speech_thread.start()
        logger.info("已啟動語音工作線程")

def speak_text(text):
    """將文本加入語音隊列"""
    if not text:
        return
    try:
        speech_queue.put(text, block=False)  # 不阻塞，如果隊列滿了就丟棄
        logger.info(f"已將文本加入隊列，當前隊列大小: {speech_queue.qsize()}")
    except Queue.Full:
        logger.warning("語音隊列已滿，丟棄新文本")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# 使用轮询传输
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   max_http_buffer_size=1 * 1024 * 1024,
                   ping_timeout=60,
                   ping_interval=25,
                   async_mode='threading')

# 添加连接管理
connected_clients = set()
frame_queue = Queue(maxsize=2)  # 限制队列大小

@socketio.on('connect')
def handle_connect():
    logger.info('客户端已连接')
    connected_clients.add(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('客户端已断开连接')
    connected_clients.discard(request.sid)

@socketio.on_error()
def error_handler(e):
    logger.error(f'Socket.IO错误: {str(e)}')

def extract_chinese(text):
    """提取中文字符"""
    logger.debug(f"提取中文字符，输入文本: {text}")
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    result = pattern.findall(text)
    extracted = ''.join(result)
    logger.debug(f"提取结果: {extracted}")
    return extracted

def process_frame(frame_data):
    """處理圖像幀，進行OCR識別"""
    logger.info("開始處理新的圖像幀")
    try:
        # 解碼base64圖像數據
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 圖像預處理
        # 1. 調整圖像大小（保持寬高比）
        height, width = frame.shape[:2]
        max_dimension = 800
        scale = max_dimension / max(height, width)
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # 2. 轉換為灰度圖
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 3. 應用高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. 自適應閾值處理
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10
        )
        
        # 5. 腐蝕和膨脹操作以去除噪點
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # 6. 對比度增強
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(binary)
        
        # 7. 使用輪廓檢測來識別文字區域
        contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 8. 創建文字區域掩碼
        text_mask = np.zeros_like(enhanced)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:  # 過濾掉不符合文字特徵的區域
                cv2.rectangle(text_mask, (x, y), (x+w, y+h), 255, -1)
        
        # 9. 應用掩碼
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=text_mask)
        
        # OCR識別
        custom_config = r'--oem 1 --psm 6 -l chi_tra --dpi 300'
        text = pytesseract.image_to_string(enhanced, config=custom_config)
        
        # 提取中文字符
        chinese_text = extract_chinese(text)
        logger.info(f"最終識別結果: {chinese_text}")
        
        # 將灰圖轉換為彩色圖像
        binary_colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # 獲取文字框信息並繪製
        boxes = pytesseract.image_to_boxes(enhanced, config=custom_config)
        for box in boxes.splitlines():
            box = box.split(' ')
            if len(box) >= 6:  # 確保有足夠的數據
                x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
                # 注意：Tesseract 的坐標系統是從左下角開始的
                cv2.rectangle(binary_colored, (x1, height - y2), (x2, height - y1), (0, 255, 0), 2)
        
        # 將處理後的圖像轉換為base64
        _, buffer = cv2.imencode('.jpg', binary_colored)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'text': chinese_text,
            'processed_image': processed_image
        }
    except Exception as e:
        logger.error(f"處理圖像幀時發生錯誤: {str(e)}")
        return {
            'text': '',
            'processed_image': ''
        }

def process_frame_worker():
    """处理帧的工作线程"""
    while True:
        try:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                try:
                    result = process_frame(frame_data)
                    
                    # 如果有識別到文字，發送結果並加入語音隊列
                    if result['text']:
                        socketio.emit('recognized_text', result)
                        speak_text(result['text'])
                except Exception as e:
                    logger.error(f"處理幀時發生錯誤: {str(e)}")
            time.sleep(0.1)  # 減少延遲，提高處理頻率
        except Exception as e:
            logger.error(f"工作線程錯誤: {str(e)}")

# 在程序啟動時啟動語音工作線程
start_speech_thread()

# 启动工作线程
threading.Thread(target=process_frame_worker, daemon=True).start()

@socketio.on('frame')
def handle_frame(frame_data):
    """处理接收到的视频帧"""
    try:
        logger.debug("收到新的視頻幀")
        if frame_queue.full():
            frame_queue.get()  # 如果队列已满，移除最旧的帧
        frame_queue.put(frame_data)
        logger.debug("已將幀數據加入隊列")
        return {'status': 'success'}  # 发送确认响应
    except Exception as e:
        logger.error(f"處理視頻幀時發生錯誤: {str(e)}")
        return {'status': 'error', 'error': str(e)}

@app.route('/')
def index():
    logger.info("访问主页")
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("启动服务器...")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
