#!/usr/bin/env python3
"""
分析 benchmark 结果文件并生成统计报告
"""
import os
import json
import glob
from pathlib import Path
from collections import defaultdict

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.getenv('BENCHMARK_RESULTS_DIR', str(BENCHMARK_DIR / 'results')))

def analyze_single_file(json_file):
    """分析单个 JSON 结果文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化统计数据
    question_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    
    # 遍历每个视频的结果
    if 'results' in data:
        for video_result in data['results']:
            choices = video_result.get('choices', [])
            correct = video_result.get('correct', [])
            
            # 统计每个问题
            for q_idx in range(len(correct)):
                if q_idx < len(choices):
                    question_stats[q_idx + 1]['total'] += 1
                    if choices[q_idx] == correct[q_idx]:
                        question_stats[q_idx + 1]['correct'] += 1
                    overall_total += 1
                    if choices[q_idx] == correct[q_idx]:
                        overall_correct += 1
    
    return question_stats, overall_correct, overall_total

def generate_report(json_file, question_stats, overall_correct, overall_total):
    """生成统计报告"""
    report_lines = []
    
    # 按问题索引排序
    for q_idx in sorted(question_stats.keys()):
        stats = question_stats[q_idx]
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            report_lines.append(f"Q{q_idx}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    # 总体准确率
    if overall_total > 0:
        overall_acc = (overall_correct / overall_total) * 100
        report_lines.append(f"总体: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    
    return '\n'.join(report_lines)

def main():
    """主程序"""
    # 查找所有 results_*.json 文件
    result_files = glob.glob(os.path.join(str(RESULTS_DIR), '**/results_*.json'), recursive=True)
    result_files.sort()
    
    print(f"找到 {len(result_files)} 个结果文件\n")
    
    # 处理每个文件
    for json_file in result_files:
        relative_path = os.path.relpath(json_file, str(RESULTS_DIR))
        
        try:
            question_stats, overall_correct, overall_total = analyze_single_file(json_file)
            report = generate_report(json_file, question_stats, overall_correct, overall_total)
            
            # 输出到标准输出
            print(f"{'='*70}")
            print(f"文件: {relative_path}")
            print(f"{'='*70}")
            print(report)
            print()
            
        except Exception as e:
            print(f"❌ 处理失败: {relative_path}")
            print(f"   错误: {str(e)}\n")

if __name__ == '__main__':
    main()
