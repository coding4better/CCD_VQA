#!/usr/bin/env python3
"""
为所有候选方案生成筛选结果集合

功能：
  1. 读取 04_candidate_thresholds.json（前15个候选方案）
  2. 对每个方案应用阈值，生成独立的筛选结果文件
  3. 生成汇总对比表格
  
输出：
  - 05_scheme_{i}_C{c}_D{d}.json — 每个方案的详细筛选结果
  - 05_schemes_comparison.json — 所有方案的对比汇总
  - 05_schemes_summary.csv — 简洁的表格对比
"""

import os
import json
import pandas as pd
import numpy as np

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'threshold_analysis')

METRICS_FILE = os.path.join(OUTPUT_DIR, '00_raw_metrics.csv')
CANDIDATES_FILE = os.path.join(OUTPUT_DIR, '04_candidate_thresholds.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载原始指标和候选方案"""
    if not os.path.exists(METRICS_FILE):
        print(f"✗ 文件不存在: {METRICS_FILE}")
        return None, None
    
    if not os.path.exists(CANDIDATES_FILE):
        print(f"✗ 文件不存在: {CANDIDATES_FILE}")
        return None, None
    
    df = pd.read_csv(METRICS_FILE)
    
    with open(CANDIDATES_FILE, 'r') as f:
        candidates = json.load(f)
    
    return df, candidates

def generate_scheme_result(df, scheme_id, cplx_th, dyn_th):
    """
    为单个方案生成筛选结果
    
    返回：
      - filtered_videos: 通过筛选的视频列表
      - statistics: 统计信息
    """
    # 应用阈值筛选
    mask = (df['scene_complexity'] >= cplx_th) & (df['dynamic_change'] >= dyn_th)
    filtered_df = df[mask].copy()
    
    # 按 dynamic_change 降序排列
    filtered_df = filtered_df.sort_values('dynamic_change', ascending=False)
    
    # 统计信息
    total_count = len(df)
    filtered_count = len(filtered_df)
    retention_rate = filtered_count / total_count * 100
    
    baseline_dyn = df['dynamic_change'].mean()
    baseline_cplx = df['scene_complexity'].mean()
    
    filtered_dyn = filtered_df['dynamic_change'].mean()
    filtered_cplx = filtered_df['scene_complexity'].mean()
    
    dyn_improvement = (filtered_dyn - baseline_dyn) / baseline_dyn * 100
    cplx_improvement = (filtered_cplx - baseline_cplx) / baseline_cplx * 100
    
    statistics = {
        'scheme_id': scheme_id,
        'complexity_threshold': int(cplx_th),
        'dynamic_threshold': float(dyn_th),
        'total_videos': int(total_count),
        'filtered_count': int(filtered_count),
        'retention_rate': float(retention_rate),
        'baseline': {
            'avg_dynamic': float(baseline_dyn),
            'avg_complexity': float(baseline_cplx)
        },
        'filtered': {
            'avg_dynamic': float(filtered_dyn),
            'avg_complexity': float(filtered_cplx),
            'min_dynamic': float(filtered_df['dynamic_change'].min()),
            'max_dynamic': float(filtered_df['dynamic_change'].max()),
            'min_complexity': int(filtered_df['scene_complexity'].min()),
            'max_complexity': int(filtered_df['scene_complexity'].max())
        },
        'improvement': {
            'dynamic_percent': float(dyn_improvement),
            'complexity_percent': float(cplx_improvement)
        }
    }
    
    # 视频列表
    videos = []
    for _, row in filtered_df.iterrows():
        videos.append({
            'video_name': row['video_name'],
            'accident_frame': int(row['accident_frame']),
            'scene_complexity': int(row['scene_complexity']),
            'dynamic_change': float(row['dynamic_change']),
            'window_length': int(row['window_length'])
        })
    
    return videos, statistics

def export_scheme_files(df, candidates):
    """为所有候选方案生成独立文件"""
    
    all_schemes = []
    summary_rows = []
    
    print("\n" + "="*100)
    print("为所有候选方案生成筛选结果文件")
    print("="*100 + "\n")
    
    for idx, candidate in enumerate(candidates):
        scheme_id = idx + 1
        cplx_th = candidate['complexity_threshold']
        dyn_th = candidate['dynamic_threshold']
        
        # 生成筛选结果
        videos, stats = generate_scheme_result(df, scheme_id, cplx_th, dyn_th)
        
        # 保存独立文件
        scheme_filename = f"05_scheme_{scheme_id:02d}_C{cplx_th}_D{dyn_th:.2f}.json"
        scheme_path = os.path.join(OUTPUT_DIR, scheme_filename)
        
        scheme_data = {
            'description': f'方案 #{scheme_id}: Complexity≥{cplx_th}, Dynamic≥{dyn_th:.4f}',
            'statistics': stats,
            'videos': videos
        }
        
        with open(scheme_path, 'w', encoding='utf-8') as f:
            json.dump(scheme_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 方案 #{scheme_id:2d} | C≥{cplx_th}, D≥{dyn_th:.2f} | {stats['filtered_count']:3d} 个视频 | {scheme_filename}")
        
        # 收集汇总信息
        all_schemes.append(scheme_data)
        
        summary_rows.append({
            'scheme_id': scheme_id,
            'complexity_threshold': cplx_th,
            'dynamic_threshold': dyn_th,
            'video_count': stats['filtered_count'],
            'retention_rate': stats['retention_rate'],
            'avg_dynamic': stats['filtered']['avg_dynamic'],
            'avg_complexity': stats['filtered']['avg_complexity'],
            'dynamic_improvement': stats['improvement']['dynamic_percent'],
            'complexity_improvement': stats['improvement']['complexity_percent']
        })
    
    # 导出汇总对比文件
    comparison_path = os.path.join(OUTPUT_DIR, '05_schemes_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': '所有候选方案的筛选结果汇总对比',
            'total_schemes': len(all_schemes),
            'schemes': all_schemes
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 汇总对比文件: 05_schemes_comparison.json")
    
    # 导出简洁的 CSV 对比表
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUTPUT_DIR, '05_schemes_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
    print(f"✓ 简洁对比表格: 05_schemes_summary.csv")
    
    return summary_df

def print_comparison_table(summary_df):
    """打印简洁的对比表格"""
    print("\n" + "="*100)
    print("方案对比总览")
    print("="*100 + "\n")
    
    print(f"{'方案':<6} {'Cplx':<6} {'Dyn':<8} {'视频数':<8} {'保留率':<8} "
          f"{'平均Cplx':<10} {'平均Dyn':<10} {'Cplx↑':<8} {'Dyn↑':<8}")
    print("-"*100)
    
    for _, row in summary_df.iterrows():
        print(f"#{row['scheme_id']:<5} {row['complexity_threshold']:<6} {row['dynamic_threshold']:<8.2f} "
              f"{row['video_count']:<8} {row['retention_rate']:<7.1f}% "
              f"{row['avg_complexity']:<10.2f} {row['avg_dynamic']:<10.4f} "
              f"{row['complexity_improvement']:<7.1f}% {row['dynamic_improvement']:<7.1f}%")
    
    print("\n")

def highlight_recommendations(summary_df):
    """根据不同目标高亮推荐方案"""
    print("="*100)
    print("推荐方案（按不同目标）")
    print("="*100 + "\n")
    
    # 目标1: 样本量接近 200
    closest_200 = summary_df.iloc[(summary_df['video_count'] - 200).abs().argsort()[:3]]
    print("🎯 目标样本量 ~200 条：")
    for _, row in closest_200.iterrows():
        print(f"   方案 #{int(row['scheme_id'])}: C≥{int(row['complexity_threshold'])}, D≥{row['dynamic_threshold']:.2f} "
              f"→ {int(row['video_count'])} 条 (偏差 {int(row['video_count']) - 200:+d})")
    
    # 目标2: 最高质量提升
    top_quality = summary_df.nlargest(3, 'dynamic_improvement')
    print("\n📈 最高 Dynamic 提升：")
    for _, row in top_quality.iterrows():
        print(f"   方案 #{int(row['scheme_id'])}: C≥{int(row['complexity_threshold'])}, D≥{row['dynamic_threshold']:.2f} "
              f"→ Dynamic 提升 {row['dynamic_improvement']:.1f}%, Complexity 提升 {row['complexity_improvement']:.1f}%")
    
    # 目标3: 平衡（样本量 + 质量）
    # 计算综合分数：样本量接近200的程度 + 质量提升
    summary_df_copy = summary_df.copy()
    summary_df_copy['distance_to_200'] = (summary_df_copy['video_count'] - 200).abs()
    summary_df_copy['composite_score'] = (
        -summary_df_copy['distance_to_200'] / 20  # 归一化距离（负值）
        + summary_df_copy['dynamic_improvement'] / 5  # 归一化 Dynamic 提升
        + summary_df_copy['complexity_improvement'] / 10  # 归一化 Complexity 提升
    )
    
    top_balanced = summary_df_copy.nlargest(3, 'composite_score')
    print("\n⚖️  平衡方案（样本量 + 质量）：")
    for _, row in top_balanced.iterrows():
        print(f"   方案 #{int(row['scheme_id'])}: C≥{int(row['complexity_threshold'])}, D≥{row['dynamic_threshold']:.2f} "
              f"→ {int(row['video_count'])} 条, Dyn↑ {row['dynamic_improvement']:.1f}%, Cplx↑ {row['complexity_improvement']:.1f}%")
    
    print("\n")

def main():
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "  候选方案筛选结果生成器".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100 + "\n")
    
    # 加载数据
    df, candidates = load_data()
    if df is None or candidates is None:
        return
    
    print(f"✓ 已加载 {len(df)} 个视频的指标数据")
    print(f"✓ 已加载 {len(candidates)} 个候选方案\n")
    
    # 生成所有方案的筛选结果文件
    summary_df = export_scheme_files(df, candidates)
    
    # 打印对比表格
    print_comparison_table(summary_df)
    
    # 高亮推荐方案
    highlight_recommendations(summary_df)
    
    print("="*100)
    print("✓ 所有方案的筛选结果已生成！")
    print("="*100)
    print("\n查看结果文件：")
    print("  - 05_scheme_XX_CX_DX.XX.json — 每个方案的详细视频列表")
    print("  - 05_schemes_comparison.json — 所有方案的汇总对比")
    print("  - 05_schemes_summary.csv — 简洁对比表格（可用 Excel 打开）")
    print("\n")

if __name__ == '__main__':
    main()
