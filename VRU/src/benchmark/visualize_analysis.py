"""
可视化脚本：展示各个模型在6个方面的表现
汇总 analysis 文件夹中的所有模型数据
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns

# 设置英文字体
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

def load_analysis_data(analysis_dir):
    """Load all analysis files"""
    data = {}
    analysis_path = Path(analysis_dir)
    
    for json_file in analysis_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
                model_name = file_data.get('model_name', json_file.stem)
                data[model_name] = file_data
                print(f"✓ Loaded: {model_name}")
        except Exception as e:
            print(f"✗ Error reading {json_file}: {e}")
    
    return data

def extract_metrics_dataframe(data):
    """Extract all metrics to DataFrame"""
    records = []
    
    for model_name, model_data in data.items():
        questions = model_data.get('questions', {})
        overall = model_data.get('overall', {})
        
        record = {
            'Model': model_name,
            'Q1': questions.get('Q1', {}).get('accuracy', 0),
            'Q2': questions.get('Q2', {}).get('accuracy', 0),
            'Q3': questions.get('Q3', {}).get('accuracy', 0),
            'Q4': questions.get('Q4', {}).get('accuracy', 0),
            'Q5': questions.get('Q5', {}).get('accuracy', 0),
            'Q6': questions.get('Q6', {}).get('accuracy', 0),
            'Overall': overall.get('accuracy', 0),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('Overall', ascending=False).reset_index(drop=True)
    return df

def plot_overall_comparison(df, output_dir):
    """绘制模型总体准确率对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(df))
    bars = ax.barh(df['Model'], df['Overall'], color=colors)
    
    # 添加数值标签
    for i, (model, overall) in enumerate(zip(df['Model'], df['Overall'])):
        ax.text(overall + 0.01, i, f'{overall:.2%}', va='center', fontweight='bold')
    
    ax.set_xlabel('Overall Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_overall_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: 01_overall_comparison.png")
    plt.close()

def plot_questions_heatmap(df, output_dir):
    """绘制6个方面的热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 准备热力图数据
    heatmap_data = df[['Model', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']].set_index('Model')
    
    # 绘制热力图
    sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                cbar_kws={'label': 'Accuracy'}, ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Model Performance Across 6 Aspects (Q1-Q6)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_questions_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: 02_questions_heatmap.png")
    plt.close()

def plot_questions_radar(df, output_dir):
    """绘制雷达图：每个模型在6个方面的表现（单独保存）"""
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    angles = np.linspace(0, 2 * np.pi, len(questions), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = sns.color_palette("husl", len(df))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        values = [row[q] for q in questions]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, color=colors[idx], label=row['Model'])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(questions, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.set_title(f"{row['Model']}\n(Overall: {row['Overall']:.2%})", 
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, linewidth=0.8)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        plt.tight_layout()
        safe_name = row['Model'].replace('/', '_').replace(' ', '_')
        plt.savefig(f'{output_dir}/03_radar_{idx+1:02d}_{safe_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ 保存: 03_radar_{idx+1:02d}_{safe_name}.png")
        plt.close()

def plot_questions_line(df, output_dir):
    """绘制折线图：显示各模型在不同问题上的表现"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    colors = sns.color_palette("husl", len(df))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[q] for q in questions]
        ax.plot(questions, values, marker='o', linewidth=2.5, 
               label=row['Model'], color=colors[idx], markersize=8)
    
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Across 6 Aspects (Line Chart)', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_questions_line.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: 04_questions_line.png")
    plt.close()

def plot_questions_boxplot(df, output_dir):
    """绘制箱线图：显示各方面的分布情况"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    data_to_plot = [df[q].values for q in questions]
    
    bp = ax.boxplot(data_to_plot, labels=questions, patch_artist=True,
                     notch=True, showmeans=True)
    
    # 设置颜色
    colors = sns.color_palette("husl", len(questions))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Model Performance Across 6 Aspects', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_questions_boxplot.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: 05_questions_boxplot.png")
    plt.close()

def plot_detailed_comparison(df, output_dir):
    """绘制详细对比：所有模型在各方面的表现"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    x = np.arange(len(questions))
    width = 0.12
    colors = sns.color_palette("husl", len(df))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[q] for q in questions]
        ax.bar(x + idx * width, values, width, label=row['Model'], color=colors[idx])
    
    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Model Performance Comparison Across 6 Aspects', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(questions)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 保存: 06_detailed_comparison.png")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate summary report"""
    report = []
    report.append("=" * 80)
    report.append("MODEL PERFORMANCE ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # 整体排名
    report.append("【Overall Performance Rankings】")
    report.append("-" * 80)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        report.append(f"{idx}. {row['Model']:<30} Overall: {row['Overall']:>7.2%}")
    report.append("")
    
    # 每个问题的最佳和最差模型
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    report.append("【Performance on Each Aspect】")
    report.append("-" * 80)
    for q in questions:
        best_idx = df[q].idxmax()
        worst_idx = df[q].idxmin()
        best_model = df.loc[best_idx, 'Model']
        worst_model = df.loc[worst_idx, 'Model']
        best_acc = df.loc[best_idx, q]
        worst_acc = df.loc[worst_idx, q]
        
        report.append(f"{q}:")
        report.append(f"  Best:  {best_model:<28} {best_acc:.2%}")
        report.append(f"  Worst: {worst_model:<28} {worst_acc:.2%}")
    report.append("")
    
    # 统计信息
    report.append("【Statistical Information】")
    report.append("-" * 80)
    report.append(f"Total Models: {len(df)}")
    report.append(f"Average Overall Accuracy: {df['Overall'].mean():.2%}")
    report.append(f"Median Overall Accuracy: {df['Overall'].median():.2%}")
    report.append(f"Std Dev Overall Accuracy: {df['Overall'].std():.4f}")
    report.append("")
    
    # 各问题的平均准确率
    report.append("【Average Accuracy by Aspect】")
    report.append("-" * 80)
    for q in questions:
        avg = df[q].mean()
        report.append(f"{q}: {avg:.2%}")
    report.append("")
    
    # 详细表格
    report.append("【Detailed Metrics Table】")
    report.append("-" * 80)
    
    # 表头
    header = f"{'Model':<30} " + " ".join([f"{q:>8}" for q in questions + ['Overall']])
    report.append(header)
    report.append("-" * 80)
    
    # 数据行
    for _, row in df.iterrows():
        values = [f"{row[q]:.2%}" for q in questions + ['Overall']]
        line = f"{row['Model']:<30} " + " ".join([f"{v:>8}" for v in values])
        report.append(line)
    
    report.append("=" * 80)
    report.append("")
    
    # 写入文件
    report_text = "\n".join(report)
    with open(f'{output_dir}/ANALYSIS_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 打印到控制台
    print("\n" + report_text)
    print("✓ Saved: ANALYSIS_REPORT.txt")
    
    return df

def main():
    """Main function"""
    current_dir = Path(__file__).parent
    analysis_dir = current_dir / 'analysis'
    output_dir = current_dir / 'visualization_output'
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MODEL PERFORMANCE VISUALIZATION")
    print("=" * 60)
    print("")
    
    # 加载数据
    print("【Loading Analysis Files】")
    data = load_analysis_data(analysis_dir)
    print(f"\n✓ Successfully loaded {len(data)} model analysis files\n")
    
    # 提取指标
    print("【Extracting Metrics】")
    df = extract_metrics_dataframe(data)
    print(f"✓ Extraction complete, data shape: {df.shape}\n")
    
    # 生成可视化
    print("【Generating Visualizations】")
    plot_overall_comparison(df, output_dir)
    plot_questions_heatmap(df, output_dir)
    plot_questions_radar(df, output_dir)
    plot_questions_line(df, output_dir)
    plot_questions_boxplot(df, output_dir)
    plot_detailed_comparison(df, output_dir)
    
    # 生成汇总报告
    print("\n【Generating Summary Report】")
    generate_summary_report(df, output_dir)
    
    # 导出 CSV
    csv_path = output_dir / 'analysis_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: analysis_data.csv")
    
    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
