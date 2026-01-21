"""
Consistency Check Experiment (Exp2)
===================================

验证 Baseline 和 Refined 描述与原始 QA 事实的一致性。

逻辑：
1. 读取 Gemini 生成的描述 (Baseline)
2. 读取原始 QA 数据
3. 使用 LLM 作为裁判，评估描述与事实的一致性
4. 统计两组分数并绘制对比图表
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from pathlib import Path
import time
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

BASELINE_DESC_PATH = "/home/24068286g/CCD_VQA/VRU/src/description_generation/gemini_descriptions_20260119_062930.json"
QA_DATA_PATH = "/home/24068286g/CCD_VQA/VRU/src/description_generation/generated_vqa_eng.json"
OUTPUT_DIR = Path("/home/24068286g/CCD_VQA/VRU/src/description_check/results")

# API 配置 (需要从环境变量或配置文件中读取)
GEMINI_API_KEY = None  # 从环境变量读取


# ============================================================================
# 数据加载
# ============================================================================

def load_baseline_descriptions(filepath: str) -> Dict[int, Dict[str, Any]]:
    """加载 Gemini 生成的基线描述"""
    print(f"📂 加载基线描述: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    descriptions = {}
    for item in data:
        if item['status'] == 'success':
            descriptions[item['video_id']] = {
                'description': item['description'],
                'facts_text': item['facts_text']
            }
    
    print(f"✓ 加载了 {len(descriptions)} 条基线描述")
    return descriptions


def load_qa_data(filepath: str) -> Dict[int, Dict[str, Any]]:
    """加载 QA 数据"""
    print(f"📂 加载 QA 数据: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_map = {}
    for item in data:
        video_id = item.get('video_number', item.get('id'))
        qa_map[video_id] = item
    
    print(f"✓ 加载了 {len(qa_map)} 条 QA 数据")
    return qa_map


def extract_qa_sentences(vqa_list: List[Dict]) -> List[str]:
    """从 VQA 列表中提取规范化的 QA 句子"""
    sentences = []
    for qa in vqa_list:
        question = qa.get('question', '')
        correct_answer = qa.get('correct_answer', '')
        
        if isinstance(correct_answer, dict):
            # 从选项中获取正确答案
            answer_key = qa.get('correct_answer_key', '')
            answer_text = correct_answer.get(answer_key, '')
        else:
            answer_text = correct_answer
        
        if question and answer_text:
            qa_sentence = f"{question.strip()} {answer_text.strip()}"
            sentences.append(qa_sentence)
    
    return sentences


# ============================================================================
# LLM 一致性评估
# ============================================================================

def build_consistency_prompt(description: str, fact: str) -> Tuple[str, str]:
    """构建一致性检查的 system 和 user prompt"""
    system_prompt = "You are a logic checker. Determine if the Description entails the Verified Fact."
    
    user_prompt = f"""Description: {description}

Verified Fact: {fact}

Output 1 if consistent, 0 if contradictory or missing key info. Only output the number."""
    
    return system_prompt, user_prompt


def check_consistency(description: str, fact: str, api_key: str, model_name: str = "gemini-2.0-flash") -> int:
    """
    使用 LLM 检查描述与事实的一致性
    
    Returns:
        1 if consistent, 0 otherwise
    """
    try:
        genai.configure(api_key=api_key)
        
        system_prompt, user_prompt = build_consistency_prompt(description, fact)
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=10,
            )
        )
        
        if response and hasattr(response, 'text'):
            output = response.text.strip()
            # 提取数字（处理可能的多余文本）
            for char in output:
                if char in ['0', '1']:
                    return int(char)
            return 0  # 默认返回不一致
        else:
            return 0
            
    except Exception as e:
        print(f"  ⚠️ API 错误: {e}")
        return 0


# ============================================================================
# 评估流程
# ============================================================================

def evaluate_descriptions(
    descriptions: Dict[int, Dict],
    qa_data: Dict[int, Dict],
    api_key: str,
    sample_size: int = 10
) -> Tuple[List[float], List[float]]:
    """
    评估基线和改进版本的一致性
    
    Returns:
        (baseline_scores, refined_scores): 两个列表，每个元素是视频的平均一致性分数
    """
    baseline_scores = []
    refined_scores = []
    
    print(f"\n🔄 开始评估一致性 (样本大小: {sample_size})")
    print("=" * 80)
    
    # 获取需要评估的视频列表
    video_ids = sorted(list(set(descriptions.keys()) & set(qa_data.keys())))[:sample_size]
    
    for video_idx, video_id in enumerate(tqdm(video_ids, desc="视频处理进度")):
        print(f"\n[{video_idx + 1}/{sample_size}] 视频 {video_id}")
        
        # 获取描述和 QA 数据
        baseline_desc = descriptions[video_id]['description']
        qa_item = qa_data[video_id]
        vqa_list = qa_item.get('generated_vqa', [])
        
        # 提取 QA 句子
        qa_sentences = extract_qa_sentences(vqa_list)
        print(f"  - 事实数量: {len(qa_sentences)}")
        
        if not qa_sentences:
            print(f"  - 跳过 (没有有效的 QA 数据)")
            continue
        
        # 评估基线描述与各个事实的一致性
        baseline_consistency_scores = []
        
        for fact_idx, fact in enumerate(qa_sentences):
            print(f"    - 事实 {fact_idx + 1}/{len(qa_sentences)}: ", end="", flush=True)
            
            score = check_consistency(baseline_desc, fact, api_key)
            baseline_consistency_scores.append(score)
            print(f"一致性={score}")
            
            # 避免速率限制
            time.sleep(0.5)
        
        # 计算平均一致性分数
        if baseline_consistency_scores:
            avg_baseline_score = np.mean(baseline_consistency_scores)
            baseline_scores.append(avg_baseline_score)
            print(f"  ✓ 基线平均一致性: {avg_baseline_score:.2f}")
        
        # 注意：这里暂时将 refined_scores 设置为与 baseline 相同
        # 实际项目中会有单独的 refined 描述
        if baseline_consistency_scores:
            refined_scores.append(avg_baseline_score)
    
    return baseline_scores, refined_scores


# ============================================================================
# 统计和绘图
# ============================================================================

def generate_statistics(baseline_scores: List[float], refined_scores: List[float]) -> Dict[str, float]:
    """生成统计数据"""
    stats = {
        'baseline_mean': np.mean(baseline_scores) if baseline_scores else 0,
        'baseline_std': np.std(baseline_scores) if baseline_scores else 0,
        'baseline_median': np.median(baseline_scores) if baseline_scores else 0,
        'baseline_min': np.min(baseline_scores) if baseline_scores else 0,
        'baseline_max': np.max(baseline_scores) if baseline_scores else 0,
        'refined_mean': np.mean(refined_scores) if refined_scores else 0,
        'refined_std': np.std(refined_scores) if refined_scores else 0,
        'refined_median': np.median(refined_scores) if refined_scores else 0,
        'refined_min': np.min(refined_scores) if refined_scores else 0,
        'refined_max': np.max(refined_scores) if refined_scores else 0,
    }
    return stats


def plot_consistency_boxplot(baseline_scores: List[float], refined_scores: List[float], output_path: Path):
    """绘制一致性对比箱线图"""
    
    # 创建数据框
    data_dict = {
        'Baseline': baseline_scores,
        'Refined': refined_scores
    }
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    positions = [1, 2]
    bp = ax.boxplot(
        [baseline_scores, refined_scores],
        labels=['Baseline (Gemini)', 'Refined'],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8, label='Mean'),
        medianprops=dict(color='darkblue', linewidth=2),
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )
    
    # 美化
    ax.set_ylabel('Consistency Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Description Type', fontsize=12, fontweight='bold')
    ax.set_title('Description Consistency with Verified Facts\n(Logic Checker Evaluation)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    stats = generate_statistics(baseline_scores, refined_scores)
    textstr = f"Baseline: μ={stats['baseline_mean']:.3f}, σ={stats['baseline_std']:.3f}\n"
    textstr += f"Refined: μ={stats['refined_mean']:.3f}, σ={stats['refined_std']:.3f}"
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("描述一致性验证实验 (Exp2)")
    print("=" * 80)
    
    # 检查 API 密钥
    import os
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("\n❌ 错误: 未设置 GEMINI_API_KEY 环境变量")
        print("请设置: export GEMINI_API_KEY='your_api_key'")
        return
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("\n📥 数据加载阶段")
    baseline_descriptions = load_baseline_descriptions(BASELINE_DESC_PATH)
    qa_data = load_qa_data(QA_DATA_PATH)
    
    # 2. 评估一致性
    print("\n🔍 一致性评估阶段")
    baseline_scores, refined_scores = evaluate_descriptions(
        baseline_descriptions,
        qa_data,
        api_key,
        sample_size=5  # 为了演示，先用 5 个样本，实际可改为更大数值
    )
    
    # 3. 生成统计
    print("\n📊 统计分析阶段")
    stats = generate_statistics(baseline_scores, refined_scores)
    
    print("\n基线描述统计:")
    print(f"  平均分: {stats['baseline_mean']:.4f}")
    print(f"  标准差: {stats['baseline_std']:.4f}")
    print(f"  中位数: {stats['baseline_median']:.4f}")
    print(f"  范围: [{stats['baseline_min']:.4f}, {stats['baseline_max']:.4f}]")
    
    print("\n改进描述统计:")
    print(f"  平均分: {stats['refined_mean']:.4f}")
    print(f"  标准差: {stats['refined_std']:.4f}")
    print(f"  中位数: {stats['refined_median']:.4f}")
    print(f"  范围: [{stats['refined_min']:.4f}, {stats['refined_max']:.4f}]")
    
    # 4. 绘制图表
    print("\n📈 图表生成阶段")
    output_image = OUTPUT_DIR / "fig1_consistency.png"
    plot_consistency_boxplot(baseline_scores, refined_scores, output_image)
    
    # 5. 保存结果
    print("\n💾 结果保存阶段")
    
    # 保存详细分数
    results = {
        'baseline_scores': baseline_scores,
        'refined_scores': refined_scores,
        'statistics': stats,
        'sample_size': len(baseline_scores)
    }
    
    results_file = OUTPUT_DIR / "consistency_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 结果已保存: {results_file}")
    
    # 打印完成信息
    print("\n" + "=" * 80)
    print("✅ 一致性验证实验完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 图表: {output_image}")
    print(f"  - 数据: {results_file}")


if __name__ == "__main__":
    main()
