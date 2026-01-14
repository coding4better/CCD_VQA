import os

# --- 路径配置 ---
# 项目根目录
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = "/home/24068286g/UString"
# 输入数据路径
CCD_ROOT_PATH = os.path.join(ROOT_DIR, 'data', 'crash')
ANNOTATION_FILE = os.path.join(CCD_ROOT_PATH, 'videos', 'Crash-1500.txt')
NPZ_DIR = os.path.join(CCD_ROOT_PATH, 'yolo_features', 'positive')

# 输出数据路径
OUTPUT_DIR = os.path.join(ROOT_DIR, 'VRU', 'output2')

# --- 三种筛选策略的输出文件 ---
# 策略1: 仅使用 Scene Complexity 筛选
FILTERED_COMPLEXITY_ONLY = os.path.join(OUTPUT_DIR, 'filtered_complexity_only.json')

# 策略2: 仅使用 Dynamic Change 筛选
FILTERED_DYNAMIC_ONLY = os.path.join(OUTPUT_DIR, 'filtered_dynamic_only.json')

# 策略3: 双维度联合筛选 (推荐)
FILTERED_COMBINED = os.path.join(OUTPUT_DIR, 'filtered_combined.json')

# 综合报告输出
REPORT_OUTPUT = os.path.join(OUTPUT_DIR, 'filtering_report.txt')

# 对比分析输出 (新增)
COMPARISON_REPORT = os.path.join(OUTPUT_DIR, 'strategy_comparison.json')


# --- 筛选标准配置 ---
"""
指标选择依据：

本研究采用两个独立的视频质量评估指标，这些指标的选择基于以下理论和实证依据：

1. Scene Complexity (场景复杂度)
   - 理论依据: 交通事故通常涉及多个参与者和复杂的道路环境
   - 计算方法: 窗口内检测到的最大物体数量
   - 研究支持: 复杂场景更可能包含丰富的交互信息和关键事件
   - 统计特征: 分布呈右偏非正态 (Mean=4.58, Std=2.80, Skewness=0.707)

2. Dynamic Change (动态变化)
   - 理论依据: 事故涉及突发的运动状态变化
   - 计算方法: 帧间特征向量距离的最大值（全局P95归一化）
   - 研究支持: 动态变化大的视频更可能包含关键时刻
   - 注意: 使用两阶段全局归一化避免局部归一化导致的区分度丧失


为什么选择这2个指标而非其他？

排除的其他候选指标：
- 速度/加速度: 需要多帧跟踪和标定，计算复杂且易受遮挡影响
- 碰撞角度: 需要精确的3D姿态估计，误差大
- 天气/光照: 与事故相关性弱，更多影响检测质量而非内容质量
- 视频时长: 与内容质量无直接关系，不同数据集差异大
- 分辨率/帧率: 技术指标而非内容指标，已由检测置信度间接反映

这2个指标的独特优势：
1. 仅依赖目标检测结果，无需额外标注或跟踪
2. 计算高效，可实时处理大规模数据集
3. 物理意义明确，易于解释和验证
4. 维度互补（空间-时间），全面评估视频质量
5. 完全基于客观特征，不受主观判断影响
"""

# 物体类别ID (基于COCO数据集标准 - 已弃用VRU相关)

# 时间窗口：以事故发生帧为中心，前后各取多少帧
# 30帧 ≈ 1秒（假设30fps），足以捕捉事故前后的关键动作
TIME_WINDOW_FRAMES = 30 

# --- 筛选阈值配置 ---
# 置信度阈值：只考虑高于此置信度的检测框
CONFIDENCE_THRESHOLD = 0.5

# 策略一：动态突变性阈值 (基于帧内平均特征的帧间距离)
# 阈值选择依据：边际收益分析法
# 注：Dynamic Change 基于所有高置信度检测的平均特征向量计算
DYNAMIC_CHANGE_THRESHOLD = 0.6

# 策略二：场景复杂度阈值 (单帧峰值物体数)
# 阈值选择依据：边际收益分析法
# 注：Scene Complexity = max(该时间窗口内任意帧的高置信度检测数)
COMPLEXITY_THRESHOLD = 6

# --- 输出配置 ---
# 已在上方定义三个策略的输出文件