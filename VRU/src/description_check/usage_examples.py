"""
使用示例：如何集成和运行一致性检查实验

此文件展示了在实际项目中如何使用 exp2_consistency_check 模块。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from datetime import datetime


def example_1_basic_usage():
    """示例 1: 基础使用 - 直接导入和使用函数"""
    print("\n" + "="*80)
    print("示例 1: 基础使用")
    print("="*80)
    
    from description_check.exp2_consistency_check import (
        load_baseline_descriptions,
        load_qa_data,
        extract_qa_sentences,
        check_consistency
    )
    
    # 加载数据
    baseline_desc = load_baseline_descriptions(
        "/home/24068286g/CCD_VQA/VRU/src/description_generation/gemini_descriptions_20260119_062930.json"
    )
    qa_data = load_qa_data(
        "/home/24068286g/CCD_VQA/VRU/src/description_generation/generated_vqa_eng.json"
    )
    
    print(f"✓ 加载了 {len(baseline_desc)} 个 Baseline 描述")
    print(f"✓ 加载了 {len(qa_data)} 个 QA 数据")
    
    # 获取第一个视频的数据
    video_id = list(baseline_desc.keys())[0]
    description = baseline_desc[video_id]['description']
    vqa_list = qa_data[video_id].get('generated_vqa', [])
    
    # 提取事实
    facts = extract_qa_sentences(vqa_list)
    print(f"\n视频 {video_id} 的事实数: {len(facts)}")
    
    if facts:
        print(f"第一个事实: {facts[0][:100]}...")


def example_2_batch_evaluation():
    """示例 2: 批量评估特定视频"""
    print("\n" + "="*80)
    print("示例 2: 批量评估")
    print("="*80)
    
    from description_check.exp2_consistency_check import (
        load_baseline_descriptions,
        load_qa_data,
        extract_qa_sentences,
        check_consistency
    )
    import os
    import time
    
    # 检查 API 密钥
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("⚠️  未设置有效的 API 密钥，跳过 API 调用")
        print("   设置方法: export GEMINI_API_KEY='your_key'")
        return
    
    # 加载数据
    baseline_desc = load_baseline_descriptions(
        "/home/24068286g/CCD_VQA/VRU/src/description_generation/gemini_descriptions_20260119_062930.json"
    )
    qa_data = load_qa_data(
        "/home/24068286g/CCD_VQA/VRU/src/description_generation/generated_vqa_eng.json"
    )
    
    # 评估前 3 个视频
    video_ids = list(baseline_desc.keys())[:3]
    
    results = []
    for video_id in video_ids:
        desc = baseline_desc[video_id]['description']
        facts = extract_qa_sentences(qa_data[video_id].get('generated_vqa', []))
        
        scores = []
        for fact in facts[:2]:  # 仅评估前 2 个事实以节省时间
            score = check_consistency(desc, fact)
            scores.append(score)
            time.sleep(0.3)
        
        avg_score = np.mean(scores) if scores else 0
        results.append({
            'video_id': video_id,
            'average_score': avg_score,
            'fact_count': len(facts)
        })
        
        print(f"视频 {video_id}: 平均分 = {avg_score:.3f}")
    
    return results


def example_3_custom_evaluation():
    """示例 3: 自定义评估流程"""
    print("\n" + "="*80)
    print("示例 3: 自定义评估")
    print("="*80)
    
    # 直接创建评估函数
    
    def custom_consistency_check(desc: str, fact: str) -> dict:
        """自定义的一致性检查，返回更详细的信息"""
        # 这里可以添加自己的逻辑
        # 例如：使用不同的 LLM、不同的 Prompt 等
        
        result = {
            'description_length': len(desc),
            'fact_length': len(fact),
            'contains_fact': fact.lower() in desc.lower()
        }
        return result
    
    desc = "这是一个关于交通事故的描述..."
    fact = "这是一个验证事实..."
    
    result = custom_consistency_check(desc, fact)
    print(f"自定义检查结果: {result}")


def example_4_result_analysis():
    """示例 4: 分析现有的评估结果"""
    print("\n" + "="*80)
    print("示例 4: 结果分析")
    print("="*80)
    
    from description_check import RESULTS_DIR
    
    # 查找最新的结果文件
    result_files = list(RESULTS_DIR.glob("consistency_evaluation_*.json"))
    
    if not result_files:
        print("❌ 未找到评估结果文件")
        print("   请先运行 exp2_consistency_check.py 或 .ipynb")
        return
    
    # 加载最新的结果
    latest_file = sorted(result_files)[-1]
    print(f"\n📂 加载结果: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 分析结果
    baseline_stats = results['baseline']['statistics']
    refined_stats = results['refined']['statistics']
    
    print(f"\nBaseline 统计:")
    print(f"  平均分: {baseline_stats['mean']:.4f}")
    print(f"  标准差: {baseline_stats['std']:.4f}")
    
    print(f"\nRefined 统计:")
    print(f"  平均分: {refined_stats['mean']:.4f}")
    print(f"  标准差: {refined_stats['std']:.4f}")
    
    print(f"\n改进幅度: {results['comparison']['improvement_percent']:+.2f}%")


def example_5_visualization():
    """示例 5: 生成自定义可视化"""
    print("\n" + "="*80)
    print("示例 5: 自定义可视化")
    print("="*80)
    
    import matplotlib.pyplot as plt
    from description_check import RESULTS_DIR
    
    # 加载结果
    result_files = list(RESULTS_DIR.glob("consistency_scores_*.csv"))
    
    if not result_files:
        print("❌ 未找到 CSV 结果文件")
        return
    
    import pandas as pd
    df = pd.read_csv(sorted(result_files)[-1])
    
    print(f"✓ 加载了 {len(df)} 条记录")
    print(f"\n数据摘要:")
    print(df.describe())
    
    # 可以进一步处理数据
    # 例如：计算改进分数
    df['improvement'] = df['refined_score'] - df['baseline_score']
    print(f"\n改进分数统计:")
    print(df['improvement'].describe())


def example_6_integration_workflow():
    """示例 6: 完整的集成工作流"""
    print("\n" + "="*80)
    print("示例 6: 完整工作流")
    print("="*80)
    
    print("""
    完整的集成工作流应该包括：
    
    1. 数据准备
       - 加载 Baseline 和 Refined 描述
       - 加载 QA 数据和验证事实
    
    2. 评估执行
       - 调用 LLM 进行一致性检查
       - 收集评估分数
    
    3. 结果分析
       - 计算统计指标
       - 对比 Baseline vs Refined
    
    4. 可视化展示
       - 绘制箱线图和其他统计图表
       - 生成论文用图表
    
    5. 报告生成
       - 保存详细结果
       - 生成摘要报告
       - 导出论文素材
    
    所有这些都已在 exp2_consistency_check.py 和 .ipynb 中实现！
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="一致性检查实验使用示例"
    )
    parser.add_argument(
        "example",
        nargs="?",
        type=int,
        default=4,
        help="要运行的示例 (1-6, 默认 4)"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_usage,
        2: example_2_batch_evaluation,
        3: example_3_custom_evaluation,
        4: example_4_result_analysis,
        5: example_5_visualization,
        6: example_6_integration_workflow,
    }
    
    if args.example in examples:
        examples[args.example]()
    else:
        print(f"❌ 示例 {args.example} 不存在")
        print(f"可用示例: {list(examples.keys())}")
        
        print("\n运行所有示例:")
        for num, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"⚠️  示例 {num} 出错: {e}")
