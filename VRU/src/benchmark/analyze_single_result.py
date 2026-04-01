#!/usr/bin/env python3
"""
分析单个结果文件，统计每个视频的6个问题的准确率
用法: python analyze_single_result.py <result_file_path>
输出: analysis/<model_name>_analysis.json
"""

import json
import sys
import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = Path(os.getenv('BENCHMARK_ANALYSIS_DIR', str(BENCHMARK_DIR / 'analysis')))

def analyze_result_file(file_path):
    """
    分析结果文件并输出统计报告
    
    Args:
        file_path: 结果文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 - {e}")
        sys.exit(1)
    
    # 获取模型名称
    model_name = data.get('model_name', 'Unknown')
    
    # 获取所有结果
    results = data.get('results', [])
    
    if not results:
        print(f"错误: 结果为空")
        sys.exit(1)
    
    # 统计每个问题的准确率
    question_stats = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    total_correct = 0
    total_questions = 0
    
    # 遍历每个视频的结果
    for video_result in results:
        choices = video_result.get('choices', [])
        correct = video_result.get('correct', [])
        
        # 遍历每个问题
        for q_idx in range(1, 7):
            if q_idx - 1 < len(choices) and q_idx - 1 < len(correct):
                is_correct = (choices[q_idx - 1] == correct[q_idx - 1])
                question_stats[q_idx].append(1 if is_correct else 0)
                total_correct += 1 if is_correct else 0
                total_questions += 1
    
    # 计算每个问题的准确率
    analysis_data = {
        'model_name': model_name,
        'source_file': file_path.name,
        'questions': {},
        'overall': {}
    }
    
    # 生成问题统计数据
    for q_idx in range(1, 7):
        if question_stats[q_idx]:
            correct_count = sum(question_stats[q_idx])
            total_count = len(question_stats[q_idx])
            accuracy = correct_count / total_count if total_count > 0 else 0
            analysis_data['questions'][f'Q{q_idx}'] = {
                'accuracy': round(accuracy, 4),
                'correct': correct_count,
                'total': total_count
            }
    
    # 生成总体统计数据
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    analysis_data['overall'] = {
        'accuracy': round(overall_accuracy, 4),
        'correct': total_correct,
        'total': total_questions
    }
    
    # 创建 analysis 目录
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # 生成输出文件名
    # 移除特殊字符和空格，用于文件名
    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    output_file = ANALYSIS_DIR / f'{safe_model_name}_analysis.json'
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"模型: {model_name}")
    print(f"结果文件: {file_path.name}")
    print(f"{'='*50}")
    
    for q_idx in range(1, 7):
        if question_stats[q_idx]:
            correct_count = sum(question_stats[q_idx])
            total_count = len(question_stats[q_idx])
            accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"Q{q_idx}: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # 总体准确率
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"{'='*50}")
    print(f"总体准确率: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
    print(f"{'='*50}")
    print(f"✓ 分析报告已保存: {output_file}\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python analyze_single_result.py <result_file_path>")
        print("例如: python analyze_single_result.py results/results_internvl3-2b-8f.json")
        sys.exit(1)
    
    result_file = sys.argv[1]
    analyze_result_file(result_file)
