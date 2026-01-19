"""
数据加载器模块 - 从CSV文件加载QA对数据
负责读取和处理QA_pair_v1_3options.csv，提取Q1-Q6的问题和答案对

核心功能：
1. 读取CSV文件，包含video_number和6组QA对
2. 将每组QA对格式化为"Fact N: [Question] Answer: [Correct Answer]."
3. 返回格式化的数据列表，包含video_id、facts_text和原始qa_data

数据格式：
- 输入：CSV文件，每行包含video_number和6组(qN_text, qN_ans_correct)
- 输出：列表，每个元素为字典 {'video_id': ..., 'facts_text': "...", 'qa_data': {...}}

代码复用说明：
本模块可与以下模块配合使用：
- /home/24068286g/UString/VRU/src/api_test_framework/inference_engine.py: 模型推理引擎
- /home/24068286g/UString/VRU/src/api_test_framework/model_factory.py: 模型加载工厂
- /home/24068286g/UString/VRU/src/benchmark/models/*_runner.py: 各类模型的Runner实现
  * gemini_runner.py: Gemini API推理
  * qwen_runner.py: Qwen2.5-VL推理
  * base_runner.py: 基础Runner接口
"""

import pandas as pd
from typing import List, Dict, Any
import os


class QADataLoader:
    """QA对数据加载器"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV数据
        
        Returns:
            pandas DataFrame
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
            
        print(f"📁 正在加载数据文件: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"✓ 成功加载 {len(self.data)} 条数据记录")
        
        return self.data
    
    def format_qa_pair(self, question: str, answer: str, fact_num: int) -> str:
        """
        格式化单个QA对
        
        Args:
            question: 问题文本
            answer: 正确答案文本
            fact_num: 事实编号 (1-6)
            
        Returns:
            格式化的字符串
        """
        # 清理问题文本中的换行符和多余空格
        question = question.strip().replace('\n', ' ')
        answer = answer.strip().replace('\n', ' ')
        
        return f"Fact {fact_num}: {question} Answer: {answer}."
    
    def extract_facts_from_row(self, row: pd.Series) -> str:
        """
        从数据行中提取并格式化所有QA对
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            格式化的facts文本（包含Q1-Q6）
        """
        facts = []
        
        # 遍历Q1到Q6
        for i in range(1, 7):
            q_text_col = f'q{i}_text'
            q_ans_col = f'q{i}_ans_correct'
            
            # 检查列是否存在且有值
            if q_text_col in row and q_ans_col in row:
                if pd.notna(row[q_text_col]) and pd.notna(row[q_ans_col]):
                    fact_text = self.format_qa_pair(
                        row[q_text_col], 
                        row[q_ans_col], 
                        i
                    )
                    facts.append(fact_text)
        
        # 用换行符连接所有facts
        return '\n'.join(facts)
    
    def extract_qa_data_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        从数据行中提取完整的QA数据（包含问题、类别、正确答案和错误选项）
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            包含所有QA信息的字典
        """
        qa_data = {}
        
        for i in range(1, 7):
            q_key = f'q{i}'
            qa_data[q_key] = {
                'text': row.get(f'q{i}_text', ''),
                'category': row.get(f'q{i}_category', ''),
                'ans_correct': row.get(f'q{i}_ans_correct', ''),
                'ans_wrong1': row.get(f'q{i}_ans_wrong1', ''),
                'ans_wrong2': row.get(f'q{i}_ans_wrong2', '')
            }
        
        return qa_data
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        处理所有数据并返回格式化的结果
        
        Returns:
            列表，每个元素包含:
            {
                'video_id': str/int,
                'facts_text': str,  # 格式化的Fact Paragraph
                'qa_data': dict     # 完整的QA数据
            }
        """
        if self.data is None:
            self.load_data()
        
        processed_data = []
        
        print(f"\n📊 正在处理数据...")
        for idx, row in self.data.iterrows():
            # 提取video_id
            video_id = row['video_number']
            
            # 格式化facts文本
            facts_text = self.extract_facts_from_row(row)
            
            # 提取完整QA数据
            qa_data = self.extract_qa_data_from_row(row)
            
            # 构建结果
            processed_item = {
                'video_id': video_id,
                'facts_text': facts_text,
                'qa_data': qa_data
            }
            
            processed_data.append(processed_item)
        
        print(f"✓ 成功处理 {len(processed_data)} 条数据")
        
        return processed_data


def load_qa_data(csv_path: str) -> List[Dict[str, Any]]:
    """
    便捷函数：加载并处理QA数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        处理后的数据列表
    """
    loader = QADataLoader(csv_path)
    return loader.process_data()


if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 80)
    print("测试 QA 数据加载器")
    print("=" * 80)
    
    # CSV文件路径
    csv_path = "/home/24068286g/UString/VRU/src/option_generate/data/QA_pair_v1_3options.csv"
    
    # 加载数据
    try:
        processed_data = load_qa_data(csv_path)
        
        # 打印前两条数据
        print("\n" + "=" * 80)
        print("前两条处理后的数据：")
        print("=" * 80)
        
        for i, item in enumerate(processed_data[:2], 1):
            print(f"\n{'='*80}")
            print(f"数据 #{i}")
            print(f"{'='*80}")
            print(f"📹 Video ID: {item['video_id']}")
            print(f"\n📝 Facts Text:")
            print("-" * 80)
            print(item['facts_text'])
            print("-" * 80)
            
            print(f"\n📊 QA Data (结构预览):")
            print(f"  包含 {len(item['qa_data'])} 个问题")
            
            # 显示第一个问题的详细信息
            q1_data = item['qa_data']['q1']
            print(f"\n  示例 - Q1 详细信息:")
            print(f"    类别: {q1_data['category']}")
            print(f"    问题: {q1_data['text'][:80]}...")
            print(f"    正确答案: {q1_data['ans_correct'][:60]}...")
            print(f"    错误选项1: {q1_data['ans_wrong1'][:60]}...")
            print(f"    错误选项2: {q1_data['ans_wrong2'][:60]}...")
        
        print("\n" + "=" * 80)
        print(f"✓ 测试完成！共加载 {len(processed_data)} 条数据")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
