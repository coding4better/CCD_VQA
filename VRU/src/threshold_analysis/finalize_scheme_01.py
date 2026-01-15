#!/usr/bin/env python3
"""
方案#1 最终列表生成器

基于方案#1 (C≥6, D≥0.70) 生成：
  1. 06_final_decision.json — 最终决策文档（含决策理由）
  2. 07_final_filtered_videos.json — 最终视频列表（JSON格式）
  3. 07_final_filtered_videos.csv — 最终视频列表（CSV格式）
  4. 07_final_filtered_videos.txt — 最终视频列表（纯文本，每行一个视频名）
"""

import os
import json
import pandas as pd
from datetime import datetime

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'threshold_analysis')

SCHEME_01_FILE = os.path.join(OUTPUT_DIR, '05_scheme_01_C6_D0.70.json')
METRICS_FILE = os.path.join(OUTPUT_DIR, '00_raw_metrics.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_scheme_01():
    """加载方案#1的筛选结果"""
    if not os.path.exists(SCHEME_01_FILE):
        print(f"✗ 文件不存在: {SCHEME_01_FILE}")
        return None
    
    with open(SCHEME_01_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_decision_document(scheme_data):
    """生成最终决策文档"""
    stats = scheme_data['statistics']
    
    decision = {
        'decision_timestamp': datetime.now().isoformat(),
        'decision_stage': 'Final Selection',
        'description': '方案#1：阈值组合的最终决策',
        
        'thresholds': {
            'scene_complexity': stats['complexity_threshold'],
            'dynamic_change': stats['dynamic_threshold'],
            'logic': 'AND'
        },
        
        'rationale': {
            'reason_1': '样本量精准：恰好200个视频，在目标范围内（150-500）',
            'reason_2': f'Complexity 提升：{stats["improvement"]["complexity_percent"]:.1f}% （基线 {stats["baseline"]["avg_complexity"]:.2f} → 筛选后 {stats["filtered"]["avg_complexity"]:.2f}）',
            'reason_3': f'Dynamic 提升：{stats["improvement"]["dynamic_percent"]:.1f}% （基线 {stats["baseline"]["avg_dynamic"]:.4f} → 筛选后 {stats["filtered"]["avg_dynamic"]:.4f}）',
            'reason_4': '分位数友好：C≥6 对应P70，D≥0.70 接近P70，便于复现和解释',
            'reason_5': '平衡性：两个指标均获得显著提升，不偏颇任何一方',
        },
        
        'statistics': {
            'total_videos': stats['total_videos'],
            'filtered_count': stats['filtered_count'],
            'retention_rate_percent': stats['retention_rate'],
            
            'baseline': {
                'avg_dynamic': stats['baseline']['avg_dynamic'],
                'avg_complexity': stats['baseline']['avg_complexity']
            },
            
            'filtered': {
                'avg_dynamic': stats['filtered']['avg_dynamic'],
                'avg_complexity': stats['filtered']['avg_complexity'],
                'min_dynamic': stats['filtered']['min_dynamic'],
                'max_dynamic': stats['filtered']['max_dynamic'],
                'min_complexity': stats['filtered']['min_complexity'],
                'max_complexity': stats['filtered']['max_complexity']
            },
            
            'improvement_percent': {
                'dynamic': stats['improvement']['dynamic_percent'],
                'complexity': stats['improvement']['complexity_percent']
            }
        },
        
        'alternatives': {
            'reason_not_selected': '相比其他方案',
            'alternative_high_quality': '方案#11 (C≥4, D≥0.85): Dynamic 提升 44.6%，但样本量偏少 (158条)',
            'alternative_high_sample': '方案#15 (C≥6, D≥0.50): 样本量多 (310条)，但质量提升较低 (14.3%)',
            'chosen_reason': '方案#1 在样本量精准性、质量双指标均衡、分位数友好性三方面最佳'
        },
        
        'output_files': {
            'videos_json': '07_final_filtered_videos.json',
            'videos_csv': '07_final_filtered_videos.csv',
            'videos_txt': '07_final_filtered_videos.txt',
            'decision_document': '06_final_decision.json'
        }
    }
    
    return decision

def export_final_lists(scheme_data, decision_doc):
    """导出最终列表（多种格式）"""
    videos = scheme_data['videos']
    
    # 1. JSON 格式
    json_output = {
        'decision': decision_doc,
        'description': f"方案#1 最终筛选结果：Complexity≥{decision_doc['thresholds']['scene_complexity']}, "
                      f"Dynamic≥{decision_doc['thresholds']['dynamic_change']}",
        'total_count': len(videos),
        'videos': videos
    }
    
    json_path = os.path.join(OUTPUT_DIR, '07_final_filtered_videos.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 最终列表 (JSON): {json_path}")
    
    # 2. CSV 格式
    videos_df = pd.DataFrame(videos)
    csv_path = os.path.join(OUTPUT_DIR, '07_final_filtered_videos.csv')
    videos_df.to_csv(csv_path, index=False)
    
    print(f"✓ 最终列表 (CSV):  {csv_path}")
    
    # 3. 纯文本格式（每行一个视频名）
    txt_path = os.path.join(OUTPUT_DIR, '07_final_filtered_videos.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# 方案#1 最终筛选结果（共{len(videos)}个视频）\n")
        f.write(f"# 阈值：Complexity ≥ {decision_doc['thresholds']['scene_complexity']}, "
                f"Dynamic ≥ {decision_doc['thresholds']['dynamic_change']}\n")
        f.write(f"# 生成时间：{decision_doc['decision_timestamp']}\n\n")
        
        for i, video in enumerate(videos, 1):
            f.write(f"{i:3d}. {video['video_name']:20s} | "
                   f"Complexity: {video['scene_complexity']:2d} | "
                   f"Dynamic: {video['dynamic_change']:.4f} | "
                   f"Accident Frame: {video['accident_frame']:3d}\n")
    
    print(f"✓ 最终列表 (TXT):  {txt_path}")
    
    # 4. 决策文档 JSON
    decision_path = os.path.join(OUTPUT_DIR, '06_final_decision.json')
    with open(decision_path, 'w', encoding='utf-8') as f:
        json.dump(decision_doc, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 决策文档:       {decision_path}")
    
    return len(videos)

def print_summary(decision_doc, count):
    """打印最终摘要"""
    print("\n" + "="*100)
    print("方案#1 最终决策总结")
    print("="*100 + "\n")
    
    th = decision_doc['thresholds']
    stats = decision_doc['statistics']
    
    print(f"✓ 最终阈值组合：Complexity ≥ {th['scene_complexity']}, Dynamic ≥ {th['dynamic_change']:.2f}\n")
    
    print(f"✓ 筛选结果：")
    print(f"    - 总视频数：{stats['total_videos']}")
    print(f"    - 筛选后：{stats['filtered_count']} 个视频 ({stats['retention_rate_percent']:.1f}%)\n")
    
    print(f"✓ 质量指标提升：")
    print(f"    - Complexity: {stats['baseline']['avg_complexity']:.2f} → {stats['filtered']['avg_complexity']:.2f} "
          f"(↑ {stats['improvement_percent']['complexity']:.1f}%)")
    print(f"    - Dynamic:    {stats['baseline']['avg_dynamic']:.4f} → {stats['filtered']['avg_dynamic']:.4f} "
          f"(↑ {stats['improvement_percent']['dynamic']:.1f}%)\n")
    
    print(f"✓ 筛选后指标范围：")
    print(f"    - Complexity: [{stats['filtered']['min_complexity']}, {stats['filtered']['max_complexity']}]")
    print(f"    - Dynamic:    [{stats['filtered']['min_dynamic']:.4f}, {stats['filtered']['max_dynamic']:.4f}]\n")
    
    print(f"✓ 输出文件（共4个）：")
    print(f"    1. 06_final_decision.json      — 决策文档（含理由和统计）")
    print(f"    2. 07_final_filtered_videos.json — 详细视频列表 (JSON)")
    print(f"    3. 07_final_filtered_videos.csv  — 简洁视频列表 (CSV，可用Excel打开)")
    print(f"    4. 07_final_filtered_videos.txt  — 纯文本视频列表（含详细信息）\n")
    
    print("="*100)
    print("✓ 最终列表已生成！")
    print("="*100 + "\n")

def main():
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "  方案#1 最终列表生成器".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100 + "\n")
    
    # 加载方案#1数据
    scheme_data = load_scheme_01()
    if scheme_data is None:
        return
    
    print(f"✓ 已加载方案#1: {scheme_data['description']}\n")
    
    # 生成决策文档
    decision_doc = generate_decision_document(scheme_data)
    
    # 导出最终列表
    count = export_final_lists(scheme_data, decision_doc)
    
    # 打印摘要
    print_summary(decision_doc, count)

if __name__ == '__main__':
    main()
