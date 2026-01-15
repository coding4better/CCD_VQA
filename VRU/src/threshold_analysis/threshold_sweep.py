#!/usr/bin/env python3
"""
阈值扫描统计脚本

功能：
  读取 threshold_analysis/00_raw_metrics.csv
  对 Scene Complexity 和 Dynamic Change 进行网格扫描
  生成二维表格：每个 (Complexity, Dynamic) 组合对应保留的视频数量
  输出 JSON 与 CSV 便于决策
"""

import os
import json
import pandas as pd
import numpy as np

# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 输出目录（threshold_analysis/threshold_analysis）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'threshold_analysis')
METRICS_FILE = os.path.join(OUTPUT_DIR, '00_raw_metrics.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metrics():
    """加载原始指标"""
    if not os.path.exists(METRICS_FILE):
        print(f"✗ 文件不存在: {METRICS_FILE}")
        return None
    return pd.read_csv(METRICS_FILE)

def threshold_sweep(df):
    """
    双维度网格扫描
    返回：(Complexity, Dynamic) → 保留数量 的字典
    """
    
    # 定义扫描范围
    complexity_range = range(4, 10)  # P25 到 P80
    dynamic_range = np.arange(0.50, 0.95, 0.05)  # 常见分位数范围
    
    results = []
    sweep_table = {}
    
    print("\n" + "="*100)
    print("双维度阈值扫描统计 (AND 逻辑)")
    print("="*100)
    print(f"\n总视频数: {len(df)}")
    print(f"基线 - Dynamic 平均: {df['dynamic_change'].mean():.4f}, Complexity 平均: {df['scene_complexity'].mean():.2f}\n")
    
    print(f"{'Complexity':<15} | ", end="")
    for dyn in dynamic_range:
        print(f"Dyn≥{dyn:.2f}  ", end="")
    print()
    print("-" * (15 + len(dynamic_range) * 10))
    
    # 逐行（Complexity）扫描
    for cplx in complexity_range:
        print(f"Cplx≥{cplx:<12} | ", end="")
        
        for dyn in dynamic_range:
            # AND 逻辑：同时满足两个条件
            mask = (df['scene_complexity'] >= cplx) & (df['dynamic_change'] >= dyn)
            count = int(mask.sum())  # ensure JSON-serializable primitives
            rate = count / len(df) * 100
            
            # 保存结果
            key = f"cplx_{cplx}_dyn_{dyn:.2f}"
            sweep_table[key] = {
                'complexity_threshold': cplx,
                'dynamic_threshold': float(dyn),
                'count': int(count),
                'percentage': float(rate)
            }
            
            results.append({
                'complexity': cplx,
                'dynamic': float(dyn),
                'count': count,
                'percentage': rate
            })
            
            # 打印简洁格式
            print(f"{count:<9} ", end="")
        
        print()
    
    print("\n")
    return pd.DataFrame(results), sweep_table

def identify_candidates(df, min_samples=150, max_samples=500):
    """
    根据样本量约束，找出候选方案
    """
    print("="*100)
    print(f"候选方案（样本量 {min_samples}-{max_samples}）")
    print("="*100)
    
    complexity_range = range(4, 10)
    dynamic_range = np.arange(0.50, 0.95, 0.05)
    
    candidates = []
    
    for cplx in complexity_range:
        for dyn in dynamic_range:
            mask = (df['scene_complexity'] >= cplx) & (df['dynamic_change'] >= dyn)
            count = int(mask.sum())  # ensure JSON-serializable primitives
            
            if min_samples <= count <= max_samples:
                dyn_avg = float(df[mask]['dynamic_change'].mean())
                cplx_avg = float(df[mask]['scene_complexity'].mean())
                dyn_lift = float((dyn_avg - df['dynamic_change'].mean()) / df['dynamic_change'].mean() * 100)
                cplx_lift = float((cplx_avg - df['scene_complexity'].mean()) / df['scene_complexity'].mean() * 100)
                
                candidates.append({
                    'complexity_threshold': cplx,
                    'dynamic_threshold': float(dyn),
                    'count': count,
                    'percentage': float(count / len(df) * 100),
                    'avg_dynamic': dyn_avg,
                    'avg_complexity': cplx_avg,
                    'dynamic_improvement': dyn_lift,
                    'complexity_improvement': cplx_lift
                })
    
    # 按样本数量排序，显示前 15 个
    candidates_sorted = sorted(candidates, key=lambda x: abs(x['count'] - 200))  # 接近 200 的优先
    
    print(f"\n找到 {len(candidates)} 个候选方案\n")
    print(f"{'方案':<5} {'Complexity':<12} {'Dynamic':<12} {'样本数':<10} {'样本率':<10} {'Cplx↑':<10} {'Dyn↑':<10}")
    print("-" * 80)
    
    for i, cand in enumerate(candidates_sorted[:15]):
        print(f"{i+1:<5} {cand['complexity_threshold']:<12} {cand['dynamic_threshold']:<12.4f} "
              f"{cand['count']:<10} {cand['percentage']:<10.1f}% {cand['complexity_improvement']:<10.1f}% {cand['dynamic_improvement']:<10.1f}%")
    
    return candidates_sorted

def export_results(sweep_df, candidates, sweep_table):
    """导出结果到 JSON 和 CSV"""
    
    output_sweep_csv = os.path.join(OUTPUT_DIR, '03_threshold_sweep_table.csv')
    output_candidates_json = os.path.join(OUTPUT_DIR, '04_candidate_thresholds.json')
    output_sweep_json = os.path.join(OUTPUT_DIR, '03_threshold_sweep.json')
    
    # 导出扫描表为 CSV
    sweep_df.to_csv(output_sweep_csv, index=False)
    print(f"\n✓ 扫描表已导出: {output_sweep_csv}")
    
    # 导出候选方案为 JSON
    with open(output_candidates_json, 'w') as f:
        json.dump(candidates[:15], f, indent=2)
    print(f"✓ 候选方案已导出: {output_candidates_json}")
    
    # 导出完整扫描为 JSON
    with open(output_sweep_json, 'w') as f:
        json.dump(sweep_table, f, indent=2)
    print(f"✓ 完整扫描结果已导出: {output_sweep_json}")

def main():
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "  阈值双维度扫描统计".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    
    df = load_metrics()
    if df is None:
        return
    
    sweep_df, sweep_table = threshold_sweep(df)
    candidates = identify_candidates(df, min_samples=150, max_samples=500)
    export_results(sweep_df, candidates, sweep_table)
    
    print("\n" + "="*100)
    print("✓ 统计完成！")
    print("="*100)
    print("\n建议查看以下文件做决策：")
    print("  1. 03_threshold_sweep_table.csv — 完整扫描表")
    print("  2. 04_candidate_thresholds.json — 前15个最佳方案")
    print("\n")

if __name__ == '__main__':
    main()
