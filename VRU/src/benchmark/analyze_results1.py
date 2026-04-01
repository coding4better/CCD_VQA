#!/usr/bin/env python3
"""
Benchmark 结果统计和分析
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

RESULTS_DIR = '/home/24068286g/UString/VRU/src/benchmark'


def load_all_results() -> Dict[str, dict]:
    """加载所有模型的结果文件"""
    all_results = {}
    
    for json_file in Path(RESULTS_DIR).glob('results_*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data.get('model_name', json_file.stem.replace('results_', ''))
                all_results[model_name] = data
        except Exception as e:
            print(f"❌ 加载 {json_file} 失败: {e}")
    
    return all_results


def print_summary(all_results: Dict[str, dict]):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("📊 Benchmark 结果摘要")
    print("="*80)
    
    if not all_results:
        print("❌ 未找到任何结果文件")
        return
    
    # 按准确率排序
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get('overall_accuracy', 0),
        reverse=True
    )
    
    print(f"\n{'模型名':<30} {'准确率':<12} {'正确数':<12} {'总题数':<12} {'视频数':<8}")
    print("-"*80)
    
    for model_name, result in sorted_models:
        acc = result.get('overall_accuracy', 0)
        correct = result.get('total_correct', 0)
        total = result.get('total_questions', 0)
        num_videos = result.get('num_videos', 0)
        
        print(f"{model_name:<30} {acc:>10.2%}  {correct:>10d}  {total:>10d}  {num_videos:>6d}")
    
    print("="*80)


def export_csv(all_results: Dict[str, dict], output_path: str = None):
    """导出为 CSV 格式"""
    if output_path is None:
        output_path = Path(RESULTS_DIR) / "benchmark_summary.csv"
    
    rows = []
    for model_name, result in all_results.items():
        rows.append({
            'Model': model_name,
            'Accuracy': result.get('overall_accuracy', 0),
            'Correct': result.get('total_correct', 0),
            'Total': result.get('total_questions', 0),
            'Videos': result.get('num_videos', 0),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Accuracy', ascending=False)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ CSV 已导出: {output_path}")
    return df


def compare_models(all_results: Dict[str, dict]):
    """模型对比分析"""
    print("\n" + "="*80)
    print("🔍 模型对比分析")
    print("="*80)
    
    if len(all_results) < 2:
        print("⚠️ 至少需要 2 个模型进行对比")
        return
    
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get('overall_accuracy', 0),
        reverse=True
    )
    
    best_model, best_result = sorted_models[0]
    best_acc = best_result.get('overall_accuracy', 0)
    
    print(f"\n🏆 最佳模型: {best_model} ({best_acc:.2%})")
    
    print("\n相对性能差距:")
    for model_name, result in sorted_models[1:]:
        acc = result.get('overall_accuracy', 0)
        gap = (best_acc - acc) * 100
        print(f"  {model_name:<30}: 落后 {gap:.2f}% 个百分点")


def detail_analysis(all_results: Dict[str, dict], model_name: str):
    """单个模型的详细分析"""
    if model_name not in all_results:
        print(f"❌ 模型 {model_name} 未找到")
        return
    
    result = all_results[model_name]
    
    print(f"\n" + "="*80)
    print(f"📈 {model_name} 详细分析")
    print("="*80)
    
    print(f"\n整体指标:")
    print(f"  准确率: {result.get('overall_accuracy', 0):.2%}")
    print(f"  正确题数: {result.get('total_correct', 0)}")
    print(f"  总题数: {result.get('total_questions', 0)}")
    print(f"  视频数: {result.get('num_videos', 0)}")
    
    # 按视频统计
    video_results = result.get('results', [])
    video_accs = [v.get('accuracy', 0) for v in video_results]
    
    if video_accs:
        print(f"\n视频级准确率:")
        print(f"  最高: {max(video_accs):.2%}")
        print(f"  最低: {min(video_accs):.2%}")
        print(f"  平均: {sum(video_accs)/len(video_accs):.2%}")
    
    # 找出最难的视频
    sorted_videos = sorted(video_results, key=lambda x: x.get('accuracy', 0))
    print(f"\n最难的 5 个视频:")
    for i, v in enumerate(sorted_videos[:5], 1):
        acc = v.get('accuracy', 0)
        correct = v.get('correct_count', 0)
        total = v.get('num_questions', 0)
        print(f"  {i}. {v.get('video_number')} - {acc:.2%} ({correct}/{total})")


if __name__ == '__main__':
    print("🚀 加载 Benchmark 结果...")
    all_results = load_all_results()
    
    if all_results:
        print_summary(all_results)
        compare_models(all_results)
        
        try:
            export_csv(all_results)
        except ImportError:
            print("⚠️ pandas 未安装，跳过 CSV 导出")
        
        # 对每个模型进行详细分析
        for model_name in list(all_results.keys())[:3]:  # 仅分析前 3 个
            detail_analysis(all_results, model_name)
    else:
        print("❌ 未找到任何结果文件，请先运行 run_benchmark.py")
