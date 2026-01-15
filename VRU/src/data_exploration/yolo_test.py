import cv2
import os
from ultralytics import YOLO

# 初始化模型
model = YOLO('yolov8n.pt')

# 视频输入路径
video_dir = r'F:\data\CarCrash\videos\Crash-1500'
output_dir = r'F:\data\CarCrash\output'
os.makedirs(output_dir, exist_ok=True)
width = 1280
height = 720
# 遍历所有视频文件
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width != 1280 and height != 720:
            print(f"视频尺寸不匹配: {video_file}")
        # 打印视频帧尺寸信息
        print(f"处理视频: {video_file}")
        print(f"帧尺寸: {width}x{height}")
        print(f"帧率: {fps} FPS")
        print("-" * 50)
        
        # 定义输出视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, video_file), fourcc, fps, (width, height))
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # 目标检测和追踪
                results = model.track(frame, persist=True)
                annotated_frame = results[0].plot()
                
                # 写入输出视频
                out.write(annotated_frame)
            else:
                break
        
        cap.release()
        out.release()