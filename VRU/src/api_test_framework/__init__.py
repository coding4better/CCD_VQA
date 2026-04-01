"""
多模型视频理解评估框架
支持8个模型（4个API + 4个开源），T4 GPU适配，内存管理
"""

from .config import MultiModelConfig, ModelConfig
from .video_processor import load_video_frames, is_valid_frame
from .model_factory import ModelFactory
from .inference_engine import InferenceEngine
from .analyzer import SimilarityAnalyzer, ResultVisualizer, ResultAnalyzer
from .pipeline import process_video_batch

__version__ = "1.0.0"
__all__ = [
    "MultiModelConfig",
    "ModelConfig",
    "load_video_frames",
    "is_valid_frame",
    "ModelFactory",
    "InferenceEngine",
    "SimilarityAnalyzer",
    "ResultVisualizer",
    "ResultAnalyzer",
    "process_video_batch",
]
