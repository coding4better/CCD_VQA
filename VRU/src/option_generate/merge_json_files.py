"""
将 generated_2options.json 中的 2_options 合并到 generated_options_345.json 中
生成包含 2、3、4、5 选项版本的完整文件
"""

import json
from pathlib import Path

def merge_json_files():
    script_dir = Path(__file__).parent
    
    # 文件路径
    file_2options = script_dir / 'data' / 'generated_2options.json'
    file_345options = script_dir / 'data' / 'generated_options_345.json'
    output_file = script_dir / 'data' / 'generated_options_2345.json'
    
    print("=" * 60)
    print("合并JSON文件: 2选项 + 345选项 -> 2345选项")
    print("=" * 60)
    
    # 加载两个JSON文件
    print(f"\n加载 {file_2options.name}...")
    with open(file_2options, 'r', encoding='utf-8') as f:
        data_2 = json.load(f)
    print(f"  记录数: {len(data_2)}")
    
    print(f"\n加载 {file_345options.name}...")
    with open(file_345options, 'r', encoding='utf-8') as f:
        data_345 = json.load(f)
    print(f"  记录数: {len(data_345)}")
    
    # 将 2options 数据按 (video_id, q_id) 索引
    data_2_dict = {}
    for record in data_2:
        if 'video_id' in record and 'q_id' in record:
            key = (record['video_id'], record['q_id'])
            data_2_dict[key] = record
    
    # 合并数据
    print("\n合并数据...")
    merged_count = 0
    missing_count = 0
    
    for record in data_345:
        if 'video_id' not in record or 'q_id' not in record:
            continue
            
        key = (record['video_id'], record['q_id'])
        
        if key in data_2_dict:
            # 获取2选项的数据
            record_2 = data_2_dict[key]
            if 'results_by_num_options' in record_2 and '2_options' in record_2['results_by_num_options']:
                # 将2_options添加到345记录中
                if 'results_by_num_options' not in record:
                    record['results_by_num_options'] = {}
                record['results_by_num_options']['2_options'] = record_2['results_by_num_options']['2_options']
                merged_count += 1
        else:
            missing_count += 1
            print(f"  警告: video {record['video_id']} q{record['q_id']} 在2options中未找到")
    
    print(f"\n合并完成:")
    print(f"  成功合并: {merged_count} 条记录")
    print(f"  未找到匹配: {missing_count} 条记录")
    
    # 保存合并后的文件
    print(f"\n保存到 {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_345, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存")
    
    # 验证
    print("\n验证合并结果...")
    with open(output_file, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    # 检查第一条有效记录
    for record in merged_data:
        if 'results_by_num_options' in record:
            options = list(record['results_by_num_options'].keys())
            print(f"  第一条记录包含的选项版本: {sorted(options)}")
            break
    
    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)

if __name__ == '__main__':
    merge_json_files()
