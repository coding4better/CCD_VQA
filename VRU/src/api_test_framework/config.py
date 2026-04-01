"""
配置管理模块
定义模型配置和全局配置
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import torch
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    type: str  # "api" or "local"
    size: str  # "3B", "7B", etc.
    max_frames: int = 20  # 最大帧数
    target_fps: float = 2.0  # 目标采样 FPS
    enabled: bool = True


class MultiModelConfig:
    """全局配置管理"""
    
    def __init__(self):
        self.task = "Dense_Captioning"  # or "VQA"
        self.output_dir = "/home/24068286g/UString/VRU/src/api_test_framework/outputs"
        self.batch_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 多 GPU 环境由 transformers 的 device_map 自动管理

        # 模型列表（优先级顺序）
        self.models = {
            # ===== API 模型 =====
            "qwen_vl": ModelConfig(
                name="qwen_vl", type="api", size="3B", max_frames=20, target_fps=2.0
            ),
            "gpt_4o_mini": ModelConfig(
                name="gpt_4o_mini", type="api", size="small", max_frames=20, target_fps=2.0
            ),
            "gemini_2_0_flash": ModelConfig(
                name="gemini_2_0_flash", type="api", size="small", max_frames=20, target_fps=2.0
            ),
            # ===== 开源模型（需要显存）=====
            "internvl_1b": ModelConfig(
                name="internvl_1b", type="local", size="1B", max_frames=20, target_fps=5.0, enabled=False
            ),
            "internvl_2b": ModelConfig(
                name="internvl_2b", type="local", size="2B", max_frames=20, target_fps=5.0, enabled=False
            ),
            "internvl_4b": ModelConfig(
                name="internvl_4b", type="local", size="4B", max_frames=20, target_fps=5.0
            ),
            "internvl_8b": ModelConfig(
                name="internvl_8b", type="local", size="8B", max_frames=20, target_fps=5.0, enabled=False
            ),
            "internvl_best": ModelConfig(
                name="internvl_best", type="local", size="auto", max_frames=20, target_fps=5.0, enabled=False
            ),
            "qwen2_vl_7b": ModelConfig(
                name="qwen2_vl_7b", type="local", size="7B", max_frames=20, target_fps=4.0  # 减少帧数避免显存溢出
            ),
            "llava_next_video": ModelConfig(
                name="llava_next_video", type="local", size="7B", max_frames=32, target_fps=5.0
            ),
            "llava_onevision": ModelConfig(
                name="llava_onevision", type="local", size="0.5B", max_frames=32, target_fps=5.0
            ),
        }

        # API Keys（从环境变量）
        self.api_keys = {
            "qwen": os.getenv("ALI_INTERNATIONAL_KEY"),
            "openai": os.getenv("GPT_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }

    def get_enabled_models(self):
        """获取所有启用的模型"""
        return [m for m, cfg in self.models.items() if cfg.enabled]

    def get_api_models(self):
        """获取所有API模型"""
        return [m for m, cfg in self.models.items() if cfg.type == "api"]

    def get_local_models(self):
        """获取所有本地模型"""
        return [m for m, cfg in self.models.items() if cfg.type == "local"]

    def print_status(self):
        """打印配置状态"""
        print("=" * 80)
        print("🎯 多模型框架配置状态")
        print("=" * 80)
        print(f"任务类型: {self.task}")
        print(f"输出目录: {self.output_dir}")
        print(f"计算设备: {self.device}")
        if self.device == "cuda":
            try:
                n_gpu = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
                print(f"GPU 数量: {n_gpu} - {', '.join(names)}")
            except Exception:
                pass
        print()
        print("📊 API 模型:")
        for model_name in self.get_api_models():
            cfg = self.models[model_name]
            status = "✓" if cfg.enabled else "✗"
            print(f"  {status} {model_name:25s} (max_frames={cfg.max_frames})")
        print()
        print("📊 本地模型:")
        for model_name in self.get_local_models():
            cfg = self.models[model_name]
            status = "✓" if cfg.enabled else "✗"
            print(f"  {status} {model_name:25s} ({cfg.size}, max_frames={cfg.max_frames})")
        print()
        print("🔑 API Keys:")
        for key_name, value in self.api_keys.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key_name.upper():25s}: {'已设置' if value else '未设置'}")
        print("=" * 80)
