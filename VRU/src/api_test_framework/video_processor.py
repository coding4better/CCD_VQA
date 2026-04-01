"""
视频处理模块
负责视频帧加载和预处理
"""

import cv2
import numpy as np
from typing import Tuple


def load_video_frames(
    video_path: str, 
    max_frames: int = 50, 
    target_fps: int = 10
) -> np.ndarray:
    """
    加载视频并均匀采样帧。
    返回：(num_frames, H, W, 3) 的 numpy 数组

    参数说明：
    - max_frames: 最多提取多少帧（默认 50）
    - target_fps: 目标采样速率（默认 10 FPS）

    修复：正确处理帧采样间隔计算
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  📹 视频信息: {total_frames} 帧, FPS={fps:.2f}, 时长={duration:.2f}秒")

        if total_frames == 0:
            print(f"❌ 视频无有效帧")
            cap.release()
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)

        # ✅ 新的正确逻辑：
        if fps > 0:
            # 计算每隔多少帧采样一次
            frame_interval = max(1, int(fps / target_fps))
            print(f"  采样策略: 每隔 {frame_interval} 帧采样一次（目标 {target_fps} FPS）")
        else:
            # FPS 无效时，均匀分布采样
            frame_interval = max(1, total_frames // max_frames)
            print(f"  采样策略: FPS 无效，均匀采样")

        frames = []
        frame_count = 0

        # 逐帧读取视频
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔采样
            if frame_count % frame_interval == 0:
                # 检查帧是否有效
                if is_valid_frame(frame):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    print(f"    ✓ 采样帧 {frame_count}/{total_frames}")
                else:
                    print(f"    ⚠️ 第 {frame_count} 帧无效（黑帧/白帧），跳过")

            frame_count += 1

        cap.release()

        if not frames:
            print(f"❌ 未能提取任何有效帧")
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)

        result = np.array(frames, dtype=np.uint8)
        print(f"  ✅ 成功提取 {len(frames)} 个有效帧")
        return result

    except Exception as e:
        print(f"❌ 加载视频出错: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((1, 480, 640, 3), dtype=np.uint8)


def is_valid_frame(frame) -> bool:
    """检查帧是否有效（不是全黑或全白）"""
    if frame is None or frame.size == 0:
        return False

    # 计算帧的平均亮度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # 判断标准：
    # - 全黑：平均亮度 < 10
    # - 全白：平均亮度 > 245
    # - 无内容：标准差 < 5（没有像素变化）
    if mean_brightness < 10 or mean_brightness > 245 or std_brightness < 5:
        return False

    return True
