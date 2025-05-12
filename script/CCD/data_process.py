import numpy as np
import os
import cv2
import shutil

def extract_frames(video_path, output_dir, start_frame=0, num_frames=50):
    """从视频中提取帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 设置起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 保存帧图像
        frame_path = os.path.join(output_dir, f"{i:06d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    cap.release()
    return True

def process_video(videos_info, crash_info, is_train=True):
    features_dir = 'F:/data/CarCrash/vgg16_features'
    yolo_labels_dir = 'F:/data/CarCrash/yolo_labels'
    images_dir = 'F:/data/CarCrash/images'
    videos_dir = 'F:/data/CarCrash/videos/Crash-1500'  # 添加视频目录
    
    # 创建必要的目录
    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 根据是否为训练集选择相应的目录
    img_dir = os.path.join(images_dir, 'train' if is_train else 'val')
    label_dir = os.path.join(yolo_labels_dir, 'train' if is_train else 'val')
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for vid_info in videos_info:
        feature_path = os.path.join(features_dir, vid_info['feature_path'])
        if not os.path.exists(feature_path):
            print(f"特征文件未找到: {feature_path}")
            continue

        vid_id = os.path.splitext(os.path.basename(vid_info['feature_path']))[0]
        # 生成唯一的视频ID
        if vid_info['feature_path'].startswith('positive/'):
            unique_vid_id = f"pos_{vid_id}"
        elif vid_info['feature_path'].startswith('negative/'):
            unique_vid_id = f"neg_{vid_id}"
        else:
            unique_vid_id = vid_id  # 兜底

        # 创建视频帧目录
        video_frames_dir = os.path.join(img_dir, unique_vid_id)
        os.makedirs(video_frames_dir, exist_ok=True)
        
        # 判断类别
        if vid_info['feature_path'].startswith('positive/'):
            video_path = os.path.join('F:/data/CarCrash/videos/Crash-1500', f"{vid_id}.mp4")
        elif vid_info['feature_path'].startswith('negative/'):
            video_path = os.path.join('F:/data/CarCrash/videos/Normal', f"{vid_id}.mp4")
        else:
            print(f"未知类别: {vid_info['feature_path']}")
            continue

        print(f"尝试打开视频: {video_path}，存在: {os.path.exists(video_path)}")
        if not extract_frames(video_path, video_frames_dir):
            print(f"无法处理视频: {video_path}")
            continue
        
        # 从特征文件中读取数据
        data = np.load(feature_path)
        det = data['det']  # (50, 19, 6) 表示50帧，每帧19个边界框，每个边界框6个值(x1, y1, x2, y2, prob, cls)

        # 处理每一帧
        for frame_idx in range(det.shape[0]):
            frame_num = f"{frame_idx:06d}"
            
            # 创建标签文件名
            label_filename = f"{unique_vid_id}_{frame_num}.txt"
            label_path = os.path.join(label_dir, label_filename)
            
            # 获取当前帧的检测框
            frame_detections = det[frame_idx]  # (19, 6)
            
            # 获取事故帧信息
            if vid_id in crash_info:
                is_crash = crash_info[vid_id][frame_idx]
            else:
                is_crash = 0
            
            # 建议单独保存帧级事故标签
            crash_label_path = os.path.join(label_dir, f"{unique_vid_id}_{frame_num}_crash.txt")
            with open(crash_label_path, 'w') as f:
                f.write(f"{is_crash}\n")
            
            # YOLO标签文件只写检测框
            with open(label_path, 'w') as f:
                for box_idx in range(frame_detections.shape[0]):
                    x1, y1, x2, y2, prob, cls = frame_detections[box_idx]
                    
                    # 只处理置信度高的检测框
                    if prob < 0.5:
                        continue
                        
                    # 确保坐标有效
                    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 归一化坐标
                    x_center = (x1 + x2) / (2.0 * 1280)  # 假设图像宽度为1280
                    y_center = (y1 + y2) / (2.0 * 720)   # 假设图像高度为720
                    width = (x2 - x1) / 1280
                    height = (y2 - y1) / 720
                    
                    # 确保坐标在有效范围内
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                        # YOLO格式：<class> <x_center> <y_center> <width> <height>
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset_file(dataset_file):
    """处理数据集文件，返回视频信息列表"""
    videos_info = []
    with open(dataset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_path = parts[0]
                label = int(parts[1])
                videos_info.append({
                    'feature_path': feature_path,
                    'label': label
                })
    return videos_info

def create_dataset_yaml():
    """创建YOLO数据集配置文件"""
    yaml_content = """
path: F:/data/CarCrash  # 数据集根目录
train: images/train  # 训练集图像相对路径
val: images/val  # 验证集图像相对路径

# 类别名称
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
  5: person
"""
    
    with open('F:/data/CarCrash/dataset.yaml', 'w') as f:
        f.write(yaml_content)

def load_crash_annotations(crash_txt_path):
    crash_info = {}
    with open(crash_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            vidname = parts[0]
            binlabels = [int(x) for x in parts[1].split(',') if x != '']
            crash_info[vidname] = binlabels
            print(crash_info)
    return crash_info

if __name__ == "__main__":
    # 处理训练集
    train_videos = process_dataset_file('script/CCD/data/train.txt')
    crash_info = load_crash_annotations('script/CCD/data/Crash-1500.txt')
    process_video(train_videos, crash_info, is_train=True)
    
    # 处理测试集
    test_videos = process_dataset_file('script/CCD/data/test.txt')
    process_video(test_videos, crash_info, is_train=False)
    
    # 创建数据集配置文件
    create_dataset_yaml()
    print("数据集转换完成！")
    
