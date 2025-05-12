import os
import shutil
from pathlib import Path

# 定义源路径列表（根据你的实际情况修改）
source_paths = [
    "F:/data/CarCrash/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-003/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-004/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-005/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-006/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-007/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-008/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-009/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-010/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-011/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-012/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-013/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-014/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-015/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-016/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-017/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-018/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-019/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-020/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-021/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-022/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-023/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-024/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-025/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-026/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-027/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-028/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-029/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-030/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-031/CarCrash/vgg16_features",
    "F:/data/CarCrash-20250421T133057Z-032/CarCrash/vgg16_features",
]

# 定义目标路径
target_dir = "F:/data/CCD/vgg16_features"
os.makedirs(f"{target_dir}/positive", exist_ok=True)
os.makedirs(f"{target_dir}/negative", exist_ok=True)

def move_files(src_dir, target_subdir):
    """将 src_dir 下的所有文件移动到 target_subdir"""
    for file in Path(src_dir).glob("*"):
        try:
            shutil.move(str(file), f"{target_dir}/{target_subdir}/{file.name}")
        except shutil.Error as e:
            print(f"跳过重复文件: {file.name}")  # 处理同名文件冲突

# 合并所有 positive 和 negative 文件
for src_path in source_paths:
    src_positive = Path(src_path) / "positive"
    src_negative = Path(src_path) / "negative"
    
    if src_positive.exists():
        move_files(src_positive, "positive")
    if src_negative.exists():
        move_files(src_negative, "negative")

print("剪切合并完成！")