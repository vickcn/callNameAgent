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
from PIL import Image, ImageDraw, ImageFont

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

def analyze_chinese_features(region):
    """分析區域的繁體字特徵"""
    logger.debug("開始分析繁體字特徵")
    
    # 1. 筆畫密度分析
    stroke_density = cv2.countNonZero(region) / (region.shape[0] * region.shape[1])
    logger.debug(f"筆畫密度: {stroke_density:.4f}")
    
    # 2. 結構複雜度分析
    edges = cv2.Canny(region, 50, 150)
    complexity = cv2.countNonZero(edges) / (region.shape[0] * region.shape[1])
    logger.debug(f"結構複雜度: {complexity:.4f}")
    
    # 3. 方向梯度分析
    sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_density = np.sum(gradient_magnitude > 50) / (region.shape[0] * region.shape[1])
    logger.debug(f"梯度密度: {gradient_density:.4f}")
    
    # 4. 筆畫方向分布
    gradient_angle = np.arctan2(sobely, sobelx) * 180 / np.pi
    angle_hist = np.histogram(gradient_angle, bins=8, range=(-180, 180))[0]
    angle_variance = np.var(angle_hist)
    logger.debug(f"方向分布方差: {angle_variance:.4f}")
    logger.debug(f"方向分布直方圖: {angle_hist}")
    
    # 5. 筆畫粗細分析
    distance_transform = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    stroke_thickness = np.mean(distance_transform)
    logger.debug(f"筆畫粗細: {stroke_thickness:.4f}")
    
    # 綜合判斷是否為繁體字（調整閾值）
    is_chinese = (
        # 筆畫密度在合理範圍（降低下限，因為可能有較稀疏的字）
        0.15 <= stroke_density <= 0.8 and
        # 結構複雜度足夠高（降低閾值）
        complexity > 0.08 and
        # 梯度密度適中（擴大範圍）
        0.08 <= gradient_density <= 0.6 and
        # 筆畫方向分布較均勻（降低閾值）
        angle_variance > 50 and
        # 筆畫粗細適中（擴大範圍）
        0.8 <= stroke_thickness <= 4.0
    )
    
    # 記錄每個特徵的判斷結果
    logger.debug("特徵判斷結果:")
    logger.debug(f"筆畫密度判斷: {stroke_density:.4f}||{'通過' if 0.15 <= stroke_density <= 0.8 else '不通過'}")
    logger.debug(f"結構複雜度判斷: {complexity:.4f}||{'通過' if complexity > 0.08 else '不通過'}")
    logger.debug(f"梯度密度判斷: {gradient_density:.4f}||{'通過' if 0.08 <= gradient_density <= 0.6 else '不通過'}")
    logger.debug(f"方向分布判斷: {angle_variance:.4f}||{'通過' if angle_variance > 50 else '不通過'}")
    logger.debug(f"筆畫粗細判斷: {stroke_thickness:.4f}||{'通過' if 0.8 <= stroke_thickness <= 4.0 else '不通過'}")
    logger.debug(f"最終判斷結果: {'是繁體字' if is_chinese else '不是繁體字'}")
    
    return is_chinese, {
        'stroke_density': stroke_density,
        'complexity': complexity,
        'gradient_density': gradient_density,
        'angle_variance': angle_variance,
        'stroke_thickness': stroke_thickness
    }

def cv2_img_to_pil(cv2_img):
    """將OpenCV圖像轉換為PIL圖像"""
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)
    return pil_img

def pil_img_to_cv2(pil_img):
    """將PIL圖像轉換為OpenCV圖像"""
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def process_frame(frame_data):
    """處理圖像幀，進行OCR辨識"""
    try:
        logger.info("開始處理圖像幀")
        # 解碼 base64 圖像數據
        image_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 轉換為灰度圖
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 Tesseract 進行文字辨識
        logger.info("開始OCR辨識")
        custom_config = r'--oem 1 --psm 6 -l chi_tra --dpi 300'
        
        # 使用image_to_data來獲取更詳細的辨識結果，包括信心分數
        data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # 在圖像上繪製文字框和辨識結果
        n_boxes = len(data['text'])
        all_text = []

        # 轉換為PIL圖像以繪製中文
        pil_img = cv2_img_to_pil(frame)
        draw = ImageDraw.Draw(pil_img)
        
        # 加載中文字體
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/System/Library/Fonts/STHeiti Light.ttc',  # macOS 備選
            '/System/Library/Fonts/STHeiti Medium.ttc',  # macOS 備選
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux 備選
            '/usr/share/fonts/truetype/arphic/uming.ttc',  # Linux 備選
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux 備選
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                logger.info(f"成功加載字體: {font_path}")
                break
            except Exception as e:
                logger.debug(f"無法加載字體 {font_path}: {str(e)}")
                continue
        
        if font is None:
            logger.warning("無法找到任何中文字體，嘗試使用默認字體")
            try:
                font = ImageFont.load_default()
            except Exception as e:
                logger.error(f"加載默認字體失敗: {str(e)}")
                return {
                    'text': '',
                    'processed_image': ''
                }
        
        for i in range(n_boxes):
            try:
                # 只處理非空的文字
                if int(data['conf'][i]) > 0:  # 確保有信心分數
                    text = data['text'][i]
                    if not text.strip():  # 跳過空白文字
                        continue
                        
                    conf = int(data['conf'][i])  # 信心分數
                    
                    # 確保文字是有效的UTF-8編碼
                    try:
                        text = text.encode('utf-8').decode('utf-8')
                    except UnicodeError:
                        logger.warning(f"跳過無效的UTF-8文字: {text}")
                        continue
                        
                    all_text.append(text)
                    
                    # 獲取文字框座標
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # 在PIL圖像上繪製綠色框
                    draw.rectangle([(x, y), (x + w, y + h)], outline=(0, 255, 0), width=2)
                    
                    # 準備顯示的文字（辨識結果和分數）
                    display_text = f"{text}({conf}%)"
                    
                    try:
                        # 獲取文字大小
                        text_bbox = draw.textbbox((0, 0), display_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # 計算文字背景矩形的位置（在框的右上角）
                        text_x = x + w  # 從框的右邊開始
                        text_y = y      # 從框的頂部開始
                        
                        # 繪製白色背景
                        draw.rectangle(
                            [(text_x, text_y - text_height - 5),
                             (text_x + text_width + 5, text_y + 5)],
                            fill=(255, 255, 255)
                        )
                        
                        # 繪製黑色文字
                        draw.text((text_x, text_y - text_height), display_text, font=font, fill=(0, 0, 0))
                    except Exception as e:
                        logger.error(f"繪製文字時發生錯誤: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"處理單個文字框時發生錯誤: {str(e)}")
                continue  # 跳過這個文字框，繼續處理下一個
        
        # 將PIL圖像轉回OpenCV格式
        frame = pil_img_to_cv2(pil_img)
        
        # 將處理後的圖像轉換為 base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            'text': ' '.join(all_text),
            'processed_image': processed_image
        }
        logger.info(f"返回結果: {result['text']}")
        return result
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
