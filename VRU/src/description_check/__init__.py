"""
描述一致性验证模块

用于评估生成的视频描述与原始 QA 事实的一致性。
"""

__version__ = "1.0.0"
__author__ = "CCD-VQA Team"

from pathlib import Path

# 模块路径
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"

# 确保结果目录存在
RESULTS_DIR.mkdir(exist_ok=True)

__all__ = [
    "MODULE_DIR",
    "RESULTS_DIR",
]
