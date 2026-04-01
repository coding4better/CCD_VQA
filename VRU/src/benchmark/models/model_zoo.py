# 模型列表和runner工厂
import os


def _split_env_csv(value: str):
    return [item.strip() for item in value.split(',') if item.strip()]

def get_model_list():
    """返回要运行的模型列表
    
    支持的 InternVL 模型:
    - internvl2-2b, internvl2-4b, internvl2-8b
    - internvl2.5-1b, internvl2.5-2b, internvl2.5-4b, internvl2.5-8b
    - internvl3-1b, internvl3-2b, internvl3-4b, internvl3-8b
    
    显存分析:
    - 单个模型: 串行运行
    - 多个模型: 自动并行运行（显存充足）
      * LLaVA-NeXT-7B: ~16GB
      * InternVL3-4B: ~10GB
      * 并行合计: ~26GB（80GB GPU 充足）
    """
    default_models = [
        # 使用 Qwen2.5-VL-7B-Instruct 本地推理（已下载，推荐）
        'qwen2_5_vl_7b',
        # 使用 InternVL2.5-2B 本地推理（已下载，推荐）
        'internvl2.5-2b',
        # Gemini 模型（API 调用，需要设置 GEMINI_API_KEY 环境变量）
        # 'gemini-2.5-pro',
    ]

    env_models = os.getenv('BENCHMARK_MODELS', '').strip()
    if env_models:
        return _split_env_csv(env_models)
    return default_models


def get_model_runner(model_name):
    """根据模型名返回对应的runner实例"""
    if model_name == 'llava-next-7b':
        from .llava_next_runner import LLaVANextRunner
        return LLaVANextRunner(model_name)
    elif model_name == 'llava-next-video-7b':
        from .llava_next_video_runner import LLaVANextVideoRunner
        return LLaVANextVideoRunner(model_name)
    elif model_name == 'llava-onevision-7b':
        from .llava_onevision_runner import LLaVAOneVisionRunner
        return LLaVAOneVisionRunner(model_name)
    elif model_name.startswith('llava'):
        from .llava_runner import LLaVARunner
        return LLaVARunner(model_name)
    elif model_name.startswith('internvl'):
        from .internvl_runner import InternVLRunner
        return InternVLRunner(model_name)
    elif model_name.startswith('qwen'):
        from .qwen_runner import QwenRunner
        # 默认 10 帧，可通过环境变量 BENCHMARK_QWEN_FRAMES 覆盖
        num_frames = int(os.getenv('BENCHMARK_QWEN_FRAMES', '10'))
        return QwenRunner(model_name, num_frames=num_frames)
    elif model_name.startswith('gemini'):
        from .gemini_runner import GeminiRunner
        return GeminiRunner(model_name)
    else:
        from .base_runner import BaseRunner
        return BaseRunner(model_name)
