"""
主管道模块
负责协调整个推理流程
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import numpy as np

from .config import MultiModelConfig
from .video_processor import load_video_frames
from .model_factory import ModelFactory
from .inference_engine import InferenceEngine
from .analyzer import SimilarityAnalyzer


def process_video_batch(
    video_paths: List[str],
    model_names: List[str] = None,
    config: MultiModelConfig = None,
    prompt: str = None
):
    """
    批量处理视频，返回完整的结果 JSON。

    参数：
        video_paths: 视频文件路径列表
        model_names: 模型名称列表（如为 None，则使用全部启用的模型）
        config: 多模型配置对象
        prompt: 推理提示词
    """
    if config is None:
        config = MultiModelConfig()

    if model_names is None:
        model_names = [m for m, cfg in config.models.items() if cfg.enabled]

    if prompt is None:
        prompt = """Provide a detailed description of this crash video.
Use clear and complete sentences in one paragraph with appropriate traffic and crash-related terminology.
Include descriptions of weather conditions, road type and environment, the traffic configuration and vehicle or pedestrian appearance (such as clothing, color, type).
Mention vehicle speed, trajectory, movements, and behavior.
Focus on the dynamics of the collision, including vehicle movement, and final impact."""

    # 初始化引擎
    factory = ModelFactory(config)
    inference_engine = InferenceEngine(config, factory, prompt)
    analyzer = SimilarityAnalyzer()

    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)

    # 结果容器
    all_results = {
        "task": config.task,
        "timestamp": datetime.now().isoformat(),
        "total_videos": len(video_paths),
        "models_used": model_names,
        "videos": []
    }

    # 逐视频处理
    for video_path in tqdm(video_paths, desc="处理视频"):
        video_name = Path(video_path).stem
        print(f"\n🎬 处理: {video_name}")

        # 逐模型推理
        model_responses = {}
        model_errors = {}
        model_latencies = {}
        last_frames_count = 0  # 记录最后成功加载的帧数

        for model_name in model_names:
            print(f"    📊 {model_name}...", end=" ")

            # 为每个模型使用其配置的 max_frames 和 target_fps
            model_cfg = config.models[model_name]
            try:
                video_frames = load_video_frames(video_path, max_frames=model_cfg.max_frames, target_fps=model_cfg.target_fps)
                last_frames_count = len(video_frames)  # 更新帧数
                print(f"加载 {len(video_frames)} 帧, ", end="")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                model_responses[model_name] = ""
                model_errors[model_name] = str(e)
                model_latencies[model_name] = 0
                continue

            start = time.time()
            text, error = inference_engine.infer(model_name, video_frames)
            latency = time.time() - start

            model_responses[model_name] = text
            model_errors[model_name] = error
            model_latencies[model_name] = latency

            if error:
                print(f"❌ (错误: {error[:50]})")
            else:
                print(f"✓ ({latency:.1f}s, {len(text)} chars)")

        # 计算该视频的模型间相似度
        valid_responses = {k: v for k, v in model_responses.items() if v and not model_errors.get(k)}
        similarities = analyzer.compute_similarity(valid_responses) if valid_responses else {}

        # 保存该视频的结果
        video_result = {
            "video_id": video_name,
            "frames_count": last_frames_count,
            "responses": model_responses,
            "errors": model_errors,
            "latencies_seconds": model_latencies,
            "model_similarities": similarities,
            "similarity_ranking": analyzer.rank_similarities(similarities),
            "timestamp": datetime.now().isoformat()
        }
        all_results["videos"].append(video_result)

        # 绘制热力图（仅当有至少 2 个成功的响应）
        if len(valid_responses) >= 2:
            heatmap_path = Path(config.output_dir) / f"{video_name}_similarity_heatmap.png"
            analyzer.plot_heatmap(valid_responses, video_name, str(heatmap_path))

        print(f"  ✓ 相似度排名: {len(similarities)} 对")

    # 保存全局 JSON 结果
    json_path = Path(config.output_dir) / f"model_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 全部结果已保存: {json_path}")
    print(f"   总视频数: {all_results['total_videos']}")
    print(f"   模型数: {len(model_names)}")

    return all_results
