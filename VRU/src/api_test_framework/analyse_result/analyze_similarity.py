#!/usr/bin/env python3
"""
MLLM相似度分析脚本 - 无外部可视化库依赖版本
生成文本统计报告和决策建议
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_data(per_video_path, model_matrix_path):
    """加载相似度数据"""
    per_video = pd.read_csv(per_video_path)
    model_matrix = pd.read_csv(model_matrix_path, index_col=0)
    return per_video, model_matrix

def generate_statistics(per_video, model_matrix):
    """生成统计数据"""
    stats = {
        'global': {
            'mean': float(per_video['similarity'].mean()),
            'median': float(per_video['similarity'].median()),
            'std': float(per_video['similarity'].std()),
            'min': float(per_video['similarity'].min()),
            'max': float(per_video['similarity'].max()),
            'q1': float(per_video['similarity'].quantile(0.25)),
            'q3': float(per_video['similarity'].quantile(0.75)),
        },
        'by_pair': {},
        'by_video': {},
    }
    
    # 按模型对的统计
    for pair in per_video['pair'].unique():
        data = per_video[per_video['pair'] == pair]['similarity']
        stats['by_pair'][pair] = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'count': int(len(data)),
        }
    
    # 按视频的统计
    for vid in per_video['video_id'].unique():
        data = per_video[per_video['video_id'] == vid]['similarity']
        stats['by_video'][str(vid)] = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
        }
    
    return stats

def classify_similarity(per_video, high_threshold=0.30, low_threshold=0.15):
    """分类相似度"""
    per_video['category'] = per_video['similarity'].apply(
        lambda x: 'high' if x > high_threshold 
        else ('low' if x < low_threshold else 'medium')
    )
    return per_video

def generate_text_report(per_video, model_matrix):
    """生成文本报告"""
    
    high_count = (per_video['similarity'] > 0.30).sum()
    low_count = (per_video['similarity'] < 0.15).sum()
    medium_count = len(per_video) - high_count - low_count
    
    total = len(per_video)
    
    report = []
    report.append("=" * 80)
    report.append("MLLM视频描述相似度分析 - 统计报告")
    report.append("=" * 80)
    
    # 全局统计
    report.append("\n【全局相似度统计】")
    report.append("-" * 60)
    mean_sim = per_video['similarity'].mean()
    report.append(f"平均相似度:     {mean_sim:.4f}")
    report.append(f"中位数:        {per_video['similarity'].median():.4f}")
    report.append(f"标准差:        {per_video['similarity'].std():.4f}")
    report.append(f"范围:          [{per_video['similarity'].min():.4f}, {per_video['similarity'].max():.4f}]")
    
    # 分类分布
    report.append("\n【相似度分类分布】")
    report.append("-" * 60)
    report.append(f"高度相似 (> 0.30):   {high_count:3d} ({high_count/total*100:5.1f}%)")
    report.append(f"中等相似 (0.15-0.30): {medium_count:3d} ({medium_count/total*100:5.1f}%)")
    report.append(f"低度相似 (< 0.15):   {low_count:3d} ({low_count/total*100:5.1f}%)")
    
    # 模型对排序
    report.append("\n【模型对相似度排序（从高到低）】")
    report.append("-" * 60)
    pair_means = per_video.groupby('pair')['similarity'].mean().sort_values(ascending=False)
    for i, (pair, mean) in enumerate(pair_means.items(), 1):
        count = (per_video['pair'] == pair).sum()
        report.append(f"{i:2d}. {pair:50s} {mean:.4f}  (n={count})")
    
    # 模型间最相似的对
    report.append("\n【最相似的模型对（模型矩阵）】")
    report.append("-" * 60)
    pairs_list = []
    for i in range(len(model_matrix)):
        for j in range(i+1, len(model_matrix)):
            pairs_list.append({
                'model_a': model_matrix.index[i],
                'model_b': model_matrix.columns[j],
                'sim': model_matrix.iloc[i, j]
            })
    pairs_list.sort(key=lambda x: x['sim'], reverse=True)
    
    for i, p in enumerate(pairs_list[:5], 1):
        report.append(f"{i}. {p['model_a']:20s} <-> {p['model_b']:20s} : {p['sim']:.4f}")
    
    report.append("\n【最不相似的模型对】")
    report.append("-" * 60)
    for i, p in enumerate(pairs_list[-3:], 1):
        report.append(f"{i}. {p['model_a']:20s} <-> {p['model_b']:20s} : {p['sim']:.4f}")
    
    # 模型的平均"可靠性"
    report.append("\n【单个模型的可靠性评分】")
    report.append("(与其他模型的平均相似度 - 越高越'主流')")
    report.append("-" * 60)
    
    model_scores = {}
    for model_a in per_video['model_a'].unique():
        scores = per_video[per_video['model_a'] == model_a]['similarity']
        model_scores[model_a] = scores.mean()
    
    for model, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
        emoji = "⭐ 主流" if score > 0.25 else "◆ 独特"
        report.append(f"  {model:20s}: {score:.4f}  {emoji}")
    
    # 关键发现
    report.append("\n【关键发现】")
    report.append("-" * 60)
    
    if high_count / total > 0.45:
        report.append("✓ 模型输出高度一致（高相似度 > 45%）")
        report.append("  → 推荐进入'相似分支'")
        report.append("  → 策略：汇总所有描述，使用LLM总结")
    elif low_count / total > 0.45:
        report.append("✗ 模型输出差异大（低相似度 > 45%）")
        report.append("  → 推荐进入'不同分支'")
        report.append("  → 策略：排除异常模型，使用核心模型集合")
    else:
        report.append("⚖ 模型输出混合（高相似和低相似均衡）")
        report.append("  → 推荐进入'混合分支'")
        report.append("  → 策略：分层处理 + LLM总结")
    
    report.append(f"\n平均相似度 {mean_sim:.4f} 说明：")
    report.append("  • 模型之间有适度的多样性（正常范围0.20-0.35）")
    report.append("  • 这种多样性是优势，提供多角度视角")
    report.append("  • 不需要排除任何模型，应充分利用互补性")
    
    report.append("\n【推荐的后续步骤】")
    report.append("-" * 60)
    report.append("1. 选择用于LLM总结的模型（如GPT-4或Claude）")
    report.append("2. 编写详细的总结提示词（指导模型识别共识和分歧）")
    report.append("3. 对10-20个代表性视频进行试点")
    report.append("4. 人工评估试点结果的质量")
    report.append("5. 根据评估调整策略，扩展到全部250-300视频")
    report.append("6. 建立质量控制和评估基准")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def generate_json_output(per_video, model_matrix, stats):
    """生成JSON格式的结构化输出"""
    
    # 准备数据
    per_video_with_pair = per_video.copy()
    per_video_with_pair['pair'] = per_video_with_pair['model_a'] + ' vs ' + per_video_with_pair['model_b']
    
    output = {
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_videos': int(per_video['video_id'].nunique()),
            'total_pairs': int(per_video['pair'].nunique()),
            'total_comparisons': int(len(per_video)),
            'models': list(model_matrix.index),
        },
        'statistics': stats,
        'recommendations': {
            'high_threshold': 0.30,
            'low_threshold': 0.15,
            'suggested_branch': 'mixed' if ((per_video['similarity'] > 0.30).sum() / len(per_video) < 0.45 and 
                                           (per_video['similarity'] < 0.15).sum() / len(per_video) < 0.45)
                               else ('similar' if (per_video['similarity'] > 0.30).sum() / len(per_video) > 0.45
                                    else 'different'),
            'use_llm_summary': True,
            'exclude_models': [],
        }
    }
    
    return output

def main():
    import sys
    
    # 获取文件路径
    script_dir = Path(__file__).parent
    per_video_path = script_dir / 'per_video.csv'
    model_matrix_path = script_dir / 'model_matrix.csv'
    
    if not per_video_path.exists() or not model_matrix_path.exists():
        print(f"错误：找不到数据文件")
        print(f"  per_video.csv: {per_video_path}")
        print(f"  model_matrix.csv: {model_matrix_path}")
        sys.exit(1)
    
    print("加载数据...")
    per_video, model_matrix = load_data(str(per_video_path), str(model_matrix_path))
    
    print("计算统计...")
    per_video = classify_similarity(per_video)
    per_video['pair'] = per_video['model_a'] + ' vs ' + per_video['model_b']
    stats = generate_statistics(per_video, model_matrix)
    
    print("生成报告...")
    report = generate_text_report(per_video, model_matrix)
    
    # 保存文本报告
    report_path = script_dir / 'SIMILARITY_STATISTICS.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ 文本报告已保存: {report_path}")
    
    # 保存JSON输出
    json_output = generate_json_output(per_video, model_matrix, stats)
    json_path = script_dir / 'analysis_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON报告已保存: {json_path}")
    
    # 打印摘要
    print("\n" + report)

if __name__ == '__main__':
    main()
