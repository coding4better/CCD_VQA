"""
将JSON问答对和原CSV表格整合为三个QA-pair数据集CSV文件
- 版本1: 3个选项 (correct + 2 wrong)
- 版本2: 4个选项 (correct + 3 wrong)  
- 版本3: 5个选项 (correct + 4 wrong)
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def load_json_data(json_file: str) -> List[Dict]:
    """加载JSON问答对数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_data(csv_file: str) -> pd.DataFrame:
    """加载原CSV数据"""
    return pd.read_csv(csv_file, dtype=str)

def create_multiselect_row(video_id: str, csv_row: Dict, json_record: Dict, 
                          num_options: int, option_key: str) -> Dict:
    """
    为单个问题创建包含多选选项的行
    
    Args:
        video_id: 视频ID
        csv_row: 原CSV中的行数据
        json_record: JSON中的单个问答记录
        num_options: 选项数量 (3, 4, 或 5)
        option_key: JSON中的key ('3_options', '4_options', '5_options')
    
    Returns:
        包含多选选项的字典
    """
    q_id = json_record['q_id']
    
    # 提取正确答案和生成的错误选项
    correct_answer = json_record['correct_answer']
    generated_options = json_record['results_by_num_options'][option_key]['generated_options']
    
    # 为了保证一致性，我们取前(num_options-1)个生成选项作为错误项
    num_wrong = num_options - 1
    wrong_options = generated_options[:num_wrong]
    
    # 确保我们有足够的错误选项（如果生成的不足，用原CSV中的）
    csv_q_key = f'q{q_id}_ans_wrong'
    if len(wrong_options) < num_wrong:
        for i in range(1, 4):  # CSV中最多有3个错误选项
            wrong_key = f'{csv_q_key}{i}'
            if wrong_key in csv_row and pd.notna(csv_row[wrong_key]) and csv_row[wrong_key] != '':
                if len(wrong_options) < num_wrong:
                    wrong_options.append(csv_row[wrong_key])
    
    row = {
        'video_number': video_id,
        f'q{q_id}_text': json_record['question'],
        f'q{q_id}_category': json_record['category'],
        f'q{q_id}_ans_correct': correct_answer,
    }
    
    # 添加所有错误选项
    for i, wrong_ans in enumerate(wrong_options, 1):
        row[f'q{q_id}_ans_wrong{i}'] = wrong_ans if wrong_ans else ''
    
    # 补齐剩余的错误选项列（如果有）
    for i in range(len(wrong_options) + 1, num_options):
        row[f'q{q_id}_ans_wrong{i}'] = ''
    
    return row

def merge_datasets(json_file: str, csv_file: str, output_dir: str = '.'):
    """
    整合JSON和CSV数据，生成三个版本的数据集
    每个视频对应一行，将所有问题整合到同一行
    根据不同选项数量动态调整CSV列数
    
    Args:
        json_file: JSON问答对文件路径
        csv_file: 原CSV数据文件路径
        output_dir: 输出目录
    """
    # 加载数据
    print("加载数据...")
    json_data = load_json_data(json_file)
    csv_df = load_csv_data(csv_file)
    
    # 按video_id和q_id整理JSON数据
    json_dict = {}  # {video_id: {q_id: record}}
    for record in json_data:
        vid = record['video_id']
        qid = record['q_id']
        if vid not in json_dict:
            json_dict[vid] = {}
        json_dict[vid][qid] = record
    
    # 为三个版本创建数据集
    versions = [
        # ('3_options', 3, 'QA_pair_v1_3options.csv'),
        # ('4_options', 4, 'QA_pair_v2_4options.csv'),
        # ('5_options', 5, 'QA_pair_v3_5options.csv'),
        ('2_options', 2, 'QA_pair_v0_2options.csv'), 
    ]
    
    for option_key, num_options, output_file in versions:
        print(f"\n生成{num_options}选项版本: {output_file}")
        
        all_rows = []
        all_questions = set()  # 跟踪所有出现的问题ID
        
        for idx, csv_row in csv_df.iterrows():
            video_id = str(csv_row['video_number']).zfill(6)  # 确保6位格式
            
            # 检查是否有对应的JSON数据
            if video_id not in json_dict:
                print(f"  警告: video {video_id} 在JSON中未找到")
                continue
            
            # 为每个视频创建一行，包含所有问题
            row = {'video_number': video_id}
            has_any_question = False
            
            for q_id in range(1, 7):  # 6个问题
                q_key = f'q{q_id}_text'
                if q_key not in csv_row or pd.isna(csv_row[q_key]) or csv_row[q_key] == '':
                    continue  # 该问题不存在
                
                if q_id not in json_dict[video_id]:
                    print(f"  警告: video {video_id} question {q_id} 在JSON中未找到")
                    continue
                
                json_record = json_dict[video_id][q_id]
                all_questions.add(q_id)  # 记录出现的问题ID
                has_any_question = True
                
                # 添加该问题的所有数据到同一行
                row[f'q{q_id}_text'] = json_record['question']
                row[f'q{q_id}_category'] = json_record['category']
                row[f'q{q_id}_ans_correct'] = json_record['correct_answer']
                
                # 添加生成的错误选项
                generated_options = json_record['results_by_num_options'][option_key]['generated_options']
                num_wrong = num_options - 1
                
                for i in range(num_wrong):
                    if i < len(generated_options):
                        row[f'q{q_id}_ans_wrong{i+1}'] = generated_options[i]
                    else:
                        row[f'q{q_id}_ans_wrong{i+1}'] = ''
            
            # 只有当该视频有至少一个问题时才添加行
            if has_any_question:
                all_rows.append(row)
        
        # 按video_id排序
        all_rows = sorted(all_rows, key=lambda x: (x['video_number']))
        
        # 写入CSV文件
        output_path = Path(output_dir) / output_file
        if all_rows:
            # 动态确定列顺序，基于实际出现的问题
            columns = ['video_number']
            sorted_questions = sorted(all_questions)
            
            for q_id in sorted_questions:
                columns.extend([
                    f'q{q_id}_text',
                    f'q{q_id}_category', 
                    f'q{q_id}_ans_correct',
                ])
                # 根据num_options动态添加错误选项列
                for i in range(1, num_options):
                    columns.append(f'q{q_id}_ans_wrong{i}')
            
            # 只保留实际存在的列
            actual_columns = []
            for col in columns:
                if any(col in row for row in all_rows):
                    actual_columns.append(col)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=actual_columns)
                writer.writeheader()
                for row in all_rows:
                    # 填充缺失的列
                    for col in actual_columns:
                        if col not in row:
                            row[col] = ''
                    writer.writerow({col: row.get(col, '') for col in actual_columns})
            
            print(f"  ✓ 已生成 {output_path}")
            print(f"    数据行数: {len(all_rows)} 个视频")
            print(f"    问题数: {len(sorted_questions)}")
            print(f"    每个问题的选项数: {num_options}")
        else:
            print(f"  × 没有生成任何数据行")

if __name__ == '__main__':
    # 设置路径
    script_dir = Path(__file__).parent
    json_file = script_dir / 'data' / 'generated_2options.json'
    csv_file = script_dir / 'data' / 'QA_pair.csv'
    
    print("=" * 60)
    print("QA数据集整合工具")
    print("=" * 60)
    
    # 检查文件是否存在
    if not json_file.exists():
        print(f"错误: JSON文件不存在 {json_file}")
        sys.exit(1)
    if not csv_file.exists():
        print(f"错误: CSV文件不存在 {csv_file}")
        sys.exit(1)
    
    # 执行整合
    merge_datasets(str(json_file), str(csv_file), str(script_dir))
    
    print("\n" + "=" * 60)
    print("整合完成！")
    print("=" * 60)
