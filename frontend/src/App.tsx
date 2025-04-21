import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import './App.css';

// 使用默认的轮询传输
const socket = io('http://localhost:5001', {
  transports: ['polling'],
  forceNew: true
});

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resultCanvasRef = useRef<HTMLCanvasElement>(null);
  const [recognizedText, setRecognizedText] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // 检查WebSocket连接状态
    socket.on('connect', () => {
      console.log('已连接到后端服务器');
      setIsConnected(true);
      setError('');
    });

    socket.on('disconnect', () => {
      console.log('与后端服务器断开连接');
      setIsConnected(false);
      setError('与服务器断开连接');
    });

    socket.on('connect_error', (error) => {
      console.error('连接错误:', error);
      setError(`连接错误: ${error.message}`);
    });

    // 接收识别结果
    socket.on('recognized_text', (data) => {
      console.log('收到识别结果:', data);
      console.log('识别文本:', data.text);
      console.log('识别文本长度:', data.text ? data.text.length : 0);
      setRecognizedText(prev => {
        console.log('更新识别文本，原文本:', prev, '新文本:', data.text);
        return data.text;
      });
      
      // 显示处理后的图像
      if (data.processed_image && resultCanvasRef.current) {
        console.log('开始显示处理后的图像');
        const img = new Image();
        img.onload = () => {
          console.log('图像加载完成，尺寸:', img.width, 'x', img.height);
          const canvas = resultCanvasRef.current;
          if (canvas) {
            // 設置畫布尺寸與原始畫面相同
            const videoElement = videoRef.current;
            if (videoElement) {
              const scale = 0.3; // 保持與原始畫面相同的縮放比例
              canvas.width = videoElement.videoWidth * scale;
              canvas.height = videoElement.videoHeight * scale;
              const ctx = canvas.getContext('2d');
              if (ctx) {
                // 清除畫布
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                // 繪製圖像
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                console.log('图像绘制完成');
              }
            }
          }
        };
        img.onerror = (error) => {
          console.error('图像加载错误:', error);
        };
        img.src = 'data:image/jpeg;base64,' + data.processed_image;
      }
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('connect_error');
      socket.off('recognized_text');
    };
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        console.log('正在请求摄像头权限...');
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          console.log('摄像头已启动');
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('摄像头启动失败:', err);
        setError('无法访问摄像头，请确保已授予权限');
      }
    };

    startCamera();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let isProcessing = false;

    const processFrame = () => {
      if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true;
        
        // 缩小画布尺寸
        const scale = 0.3; // 缩小到原来的30%
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;
        
        // 绘制图像
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 转换为JPEG并降低质量
        const imageData = canvas.toDataURL('image/jpeg', 0.2);
        
        // 发送图像数据
        console.log('发送图像数据到服务器');
        socket.emit('frame', imageData, (response: any) => {
          isProcessing = false;
          if (response && response.error) {
            console.error('发送帧数据错误:', response.error);
          } else {
            console.log('帧数据发送成功');
          }
        });
      }
    };

    // 每5秒发送一次
    const frameInterval = setInterval(() => {
      requestAnimationFrame(processFrame);
    }, 1000);

    return () => clearInterval(frameInterval);
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>即時文字辨識系統</h1>
        {error && <div className="error-message">{error}</div>}
        {!isConnected && <div className="connection-status">正在連接伺服器...</div>}
        <div className="video-container">
          <div className="video-box">
            <h3>原始畫面</h3>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                border: '2px solid #333',
                maxWidth: '100%',
                height: 'auto'
              }}
            />
          </div>
          <div className="video-box">
            <h3>處理結果</h3>
            <canvas
              ref={resultCanvasRef}
              style={{
                border: '2px solid #333',
                maxWidth: '100%',
                height: 'auto',
                backgroundColor: 'black'
              }}
            />
          </div>
          <canvas
            ref={canvasRef}
            style={{ display: 'none' }}
          />
        </div>
      </header>
    </div>
  );
}

export default App;
