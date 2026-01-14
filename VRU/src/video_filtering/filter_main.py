#!/usr/bin/env python3
"""
视频筛选主程序 - 生成三种筛选策略的结果对比
复用 threshold_exploration_unsupervised.py 的函数
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

# 添加项目路径
ROOT_DIR = "/home/24068286g/UString"
sys.path.insert(0, os.path.join(ROOT_DIR, 'VRU', 'src', 'threshold_analysis'))

# 复用已有的函数
from threshold_exploration_unsupervised import (
    load_annotations,
    compute_metrics_with_global_normalization
)

# 配置
OUTPUT_DIR = os.path.join(ROOT_DIR, 'VRU', 'output2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPLEXITY_THRESHOLD = 6
DYNAMIC_CHANGE_THRESHOLD = 0.6

def main():
    print("\n" + "="*70)
    print("视频筛选 - 三种策略对比分析")
    print("="*70)
    
    # 复用已有函数加载数据
    print("\n📊 加载视频指标（复用threshold_exploration_unsupervised.py）...")
    df = compute_metrics_with_global_normalization()
    if df is None:
        print("✗ 数据加载失败")
        return
    
    print(f"✓ 成功分析 {len(df)} 个视频")
    
    # 计算基线统计
    baseline_stats = {
        'count': len(df),
        'avg_complexity': df['scene_complexity'].mean(),
        'avg_dynamic': df['dynamic_change'].mean()
    }
    
    # 应用三种筛选策略
    print("\n🔍 应用筛选策略...")
    
    # 策略1: 仅 Complexity
    complexity_only = df[df['scene_complexity'] >= COMPLEXITY_THRESHOLD].to_dict('records')
    
    # 策略2: 仅 Dynamic
    dynamic_only = df[df['dynamic_change'] >= DYNAMIC_CHANGE_THRESHOLD].to_dict('records')
    
    # 策略3: 双重筛选
    combined = df[(df['scene_complexity'] >= COMPLEXITY_THRESHOLD) & 
                  (df['dynamic_change'] >= DYNAMIC_CHANGE_THRESHOLD)].to_dict('records')
    
    # 保存三个列表
    print("\n💾 保存筛选结果...")
    
    with open(os.path.join(OUTPUT_DIR, 'filtered_complexity_only.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'description': f'仅使用 Scene Complexity ≥ {COMPLEXITY_THRESHOLD} 筛选',
            'threshold': COMPLEXITY_THRESHOLD,
            'total_count': len(complexity_only),
            'videos': sorted(complexity_only, key=lambda x: x['scene_complexity'], reverse=True)
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 策略1 (仅Complexity): {len(complexity_only)} 个视频")
    
    with open(os.path.join(OUTPUT_DIR, 'filtered_dynamic_only.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'description': f'仅使用 Dynamic Change ≥ {DYNAMIC_CHANGE_THRESHOLD} 筛选',
            'threshold': DYNAMIC_CHANGE_THRESHOLD,
            'total_count': len(dynamic_only),
            'videos': sorted(dynamic_only, key=lambda x: x['dynamic_change'], reverse=True)
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 策略2 (仅Dynamic): {len(dynamic_only)} 个视频")
    
    with open(os.path.join(OUTPUT_DIR, 'filtered_combined.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'description': f'双维度筛选 (Complexity ≥ {COMPLEXITY_THRESHOLD} AND Dynamic ≥ {DYNAMIC_CHANGE_THRESHOLD}) - 推荐',
            'complexity_threshold': COMPLEXITY_THRESHOLD,
            'dynamic_threshold': DYNAMIC_CHANGE_THRESHOLD,
            'total_count': len(combined),
            'videos': sorted(combined, key=lambda x: (x['scene_complexity'], x['dynamic_change']), reverse=True)
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 策略3 (双重筛选): {len(combined)} 个视频")
    
    # 生成对比报告
    def calc_stats(video_list):
        if not video_list:
            return {'count': 0, 'avg_complexity': 0, 'avg_dynamic': 0}
        complexities = [v['scene_complexity'] for v in video_list]
        dynamics = [v['dynamic_change'] for v in video_list]
        return {
            'count': len(video_list),
            'retention_rate': len(video_list) / len(df) * 100,
            'avg_complexity': np.mean(complexities),
            'avg_dynamic': np.mean(dynamics),
            'complexity_improvement': (np.mean(complexities) / baseline_stats['avg_complexity'] - 1) * 100,
            'dynamic_improvement': (np.mean(dynamics) / baseline_stats['avg_dynamic'] - 1) * 100
        }
    
    comparison = {
        'total_videos': len(df),
        'baseline': baseline_stats,
        'strategies': {
            'complexity_only': {'statistics': calc_stats(complexity_only)},
            'dynamic_only': {'statistics': calc_stats(dynamic_only)},
            'combined': {'statistics': calc_stats(combined), 'recommended': True}
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'strategy_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*70)
    print("筛选结果摘要")
    print("="*70)
    print(f"\n基线: {baseline_stats['count']}个视频")
    print(f"  平均Complexity: {baseline_stats['avg_complexity']:.2f}")
    print(f"  平均Dynamic: {baseline_stats['avg_dynamic']:.4f}")
    
    for name, strategy in [('策略1-仅Complexity', 'complexity_only'), 
                           ('策略2-仅Dynamic', 'dynamic_only'),
                           ('策略3-双重筛选⭐', 'combined')]:
        stats = comparison['strategies'][strategy]['statistics']
        print(f"\n{name}: {stats['count']}个 ({stats['retention_rate']:.1f}%)")
        print(f"  Complexity: {stats['avg_complexity']:.2f} (+{stats['complexity_improvement']:.1f}%)")
        print(f"  Dynamic: {stats['avg_dynamic']:.4f} (+{stats['dynamic_improvement']:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ 筛选完成！")
    print("="*70)


if __name__ == '__main__':
    main()
