"""
Data loader module for reading QA pairs from CSV files.
This module reads QA_pair_v1_3options.csv and extracts the Q1-Q6 question-answer pairs.

Core functionality:
1. Read the CSV file containing video_number and six QA pairs.
2. Format each QA pair as "Fact N: [Question] Answer: [Correct Answer]."
3. Return a structured list containing video_id, facts_text, and the original qa_data.

Data format:
- Input: CSV rows with video_number and six sets of (qN_text, qN_ans_correct)
- Output: A list of dictionaries with keys {'video_id': ..., 'facts_text': "...", 'qa_data': {...}}

Reuse notes:
This module can be used together with:
- /home/24068286g/UString/VRU/src/api_test_framework/inference_engine.py: model inference engine
- /home/24068286g/UString/VRU/src/api_test_framework/model_factory.py: model loading factory
- /home/24068286g/UString/VRU/src/benchmark/models/*_runner.py: runner implementations for different models
    * gemini_runner.py: Gemini API inference
    * qwen_runner.py: Qwen2.5-VL inference
    * base_runner.py: base runner interface
"""

import pandas as pd
from typing import List, Dict, Any
import os


class QADataLoader:
    """QA pair data loader."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data.
        
        Returns:
            pandas DataFrame
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_path}")
            
        print(f"Loading data file: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} records successfully")
        
        return self.data
    
    def format_qa_pair(self, question: str, answer: str, fact_num: int) -> str:
        """
        Format a single QA pair.
        
        Args:
            question: Question text
            answer: Correct answer text
            fact_num: Fact number (1-6)
            
        Returns:
            Formatted string
        """
        # Normalize whitespace in the question and answer text.
        question = question.strip().replace('\n', ' ')
        answer = answer.strip().replace('\n', ' ')
        
        return f"Fact {fact_num}: {question} Answer: {answer}."
    
    def extract_facts_from_row(self, row: pd.Series) -> str:
        """
        Extract and format all QA pairs from a data row.
        
        Args:
            row: One DataFrame row
            
        Returns:
            Formatted facts text containing Q1-Q6
        """
        facts = []
        
        # Iterate over Q1 through Q6.
        for i in range(1, 7):
            q_text_col = f'q{i}_text'
            q_ans_col = f'q{i}_ans_correct'
            
            # Only use rows where both columns exist and contain values.
            if q_text_col in row and q_ans_col in row:
                if pd.notna(row[q_text_col]) and pd.notna(row[q_ans_col]):
                    fact_text = self.format_qa_pair(
                        row[q_text_col], 
                        row[q_ans_col], 
                        i
                    )
                    facts.append(fact_text)
        
        # Join the facts with newlines.
        return '\n'.join(facts)
    
    def extract_qa_data_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract the full QA payload from a data row.
        
        Args:
            row: One DataFrame row
            
        Returns:
            Dictionary containing the full QA information
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
        Process all rows and return the formatted result.
        
        Returns:
            A list where each element contains:
            {
                'video_id': str/int,
                'facts_text': str,  # Formatted fact paragraph
                'qa_data': dict     # Full QA payload
            }
        """
        if self.data is None:
            self.load_data()
        
        processed_data = []
        
        print("\nProcessing data...")
        for idx, row in self.data.iterrows():
            # Extract the video ID.
            video_id = row['video_number']
            
            # Build the facts text.
            facts_text = self.extract_facts_from_row(row)
            
            # Build the full QA payload.
            qa_data = self.extract_qa_data_from_row(row)
            
            # Assemble the processed item.
            processed_item = {
                'video_id': video_id,
                'facts_text': facts_text,
                'qa_data': qa_data
            }
            
            processed_data.append(processed_item)
        
        print(f"Processed {len(processed_data)} rows successfully")
        
        return processed_data


def load_qa_data(csv_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function to load and process QA data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed data list
    """
    loader = QADataLoader(csv_path)
    return loader.process_data()


if __name__ == "__main__":
    # Test the data loader.
    print("=" * 80)
    print("Testing QA data loader")
    print("=" * 80)
    
    # CSV file path.
    csv_path = "/home/24068286g/UString/VRU/src/option_generate/data/QA_pair_v1_3options.csv"
    
    # Load the data.
    try:
        processed_data = load_qa_data(csv_path)
        
        # Print the first two records.
        print("\n" + "=" * 80)
        print("First two processed records:")
        print("=" * 80)
        
        for i, item in enumerate(processed_data[:2], 1):
            print(f"\n{'='*80}")
            print(f"Record #{i}")
            print(f"{'='*80}")
            print(f"📹 Video ID: {item['video_id']}")
            print(f"\n📝 Facts Text:")
            print("-" * 80)
            print(item['facts_text'])
            print("-" * 80)
            
            print(f"\nQA Data (structure preview):")
            print(f"  Contains {len(item['qa_data'])} questions")
            
            # Show details for the first question.
            q1_data = item['qa_data']['q1']
            print(f"\n  Example - Q1 details:")
            print(f"    Category: {q1_data['category']}")
            print(f"    Question: {q1_data['text'][:80]}...")
            print(f"    Correct answer: {q1_data['ans_correct'][:60]}...")
            print(f"    Wrong option 1: {q1_data['ans_wrong1'][:60]}...")
            print(f"    Wrong option 2: {q1_data['ans_wrong2'][:60]}...")
        
        print("\n" + "=" * 80)
        print(f"Test complete. Loaded {len(processed_data)} records")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
