#!/usr/bin/env python3
"""
阈值分析与优化脚本

功能：
1. 计算全量数据的指标分布
2. 基于分位数给出阈值建议
3. 阈值扫描，计算各阈值组合的召回/精度/F1
4. 找到最优阈值并导出对比结果
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
import sys
from tqdm import tqdm

# ============================================================================
# 配置与路径
# ============================================================================

ROOT_DIR = "/home/24068286g/UString"
CCD_ROOT = os.path.join(ROOT_DIR, 'data', 'crash')
VRU_ROOT = os.path.join(ROOT_DIR, 'VRU')

ANNOTATION_FILE = os.path.join(CCD_ROOT, 'videos', 'Crash-1500.txt')
NPZ_DIR = os.path.join(CCD_ROOT, 'yolo_features', 'positive')
OUTPUT_DIR = os.path.join(VRU_ROOT, 'output')
ANALYSIS_OUTPUT = os.path.join(VRU_ROOT, 'threshold_analysis')

os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)

# ============================================================================
# 第一部分: 数据加载与预处理
# ============================================================================

def load_annotations(file_path):
    """加载事故标注"""
    annotations = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    vid_id = line[:6]
                    start = line.find('[')
                    end = line.find(']')
                    if start != -1 and end != -1:
                        labels = [int(x.strip()) for x in line[start+1:end].split(',')]
                        accident_frame = labels.index(1)
                        annotations[f"{vid_id}.mp4"] = accident_frame
                except:
                    continue
    except FileNotFoundError:
        print(f"✗ 标注文件未找到: {file_path}")
    return annotations


def load_ground_truth():
    """加载人工标注的真值标签"""
    ground_truth = {}
    json_path = os.path.join(OUTPUT_DIR, 'filtered_videos_analysis.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for video in data:
            ground_truth[video['video_name']] = video.get('human_judgement', 0)
    except FileNotFoundError:
        print(f"✗ 分析文件未找到: {json_path}")
    return ground_truth


def calculate_metrics(frame_features, global_max_dist=None):
    """计算动态变化评分（全局归一化）
    
    Args:
        frame_features: 窗口内的特征序列
        global_max_dist: 全局参考最大距离（用于归一化）
    
    Returns:
        归一化后的最大距离值
    """
    if frame_features.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(frame_features[:-1] - frame_features[1:], axis=1)
    max_dist = np.max(distances) if np.max(distances) > 0 else 1e-6
    
    # 如果有全局参考值，用全局参考值归一化；否则用局部最大值
    if global_max_dist is not None and global_max_dist > 0:
        distances_norm = distances / global_max_dist
    else:
        distances_norm = distances / (max_dist + 1e-6)
    
    return float(np.max(distances_norm))


def calculate_complexity(detections_list):
    """计算场景复杂度"""
    max_objs = 0
    for frame_dets in detections_list:
        if frame_dets.size > 0:
            max_objs = max(max_objs, frame_dets.shape[0])
    return max_objs


def check_vru_interaction(detections_list, vru_ids, car_ids):
    """检测VRU交互"""
    for frame_dets in detections_list:
        if frame_dets.size > 0:
            classes = set(int(obj[5]) for obj in frame_dets)
            if (classes & vru_ids) and (classes & car_ids):
                return True
    return False


# ============================================================================
# 第二部分: 全量数据分析
# ============================================================================

def compute_all_metrics(max_videos=None):
    """计算所有视频的指标分布（两阶段：先计算全局参考值，再归一化）"""
    
    print("\n" + "="*70)
    print("第一阶段: 计算全量视频指标分布（全局归一化）")
    print("="*70)
    
    annotations = load_annotations(ANNOTATION_FILE)
    ground_truth = load_ground_truth()
    
    if not annotations:
        print("✗ 无法加载标注")
        return None, None
    
    # 配置
    VRU_IDS = {0, 1, 3}
    CAR_IDS = {2, 5, 7}
    CONF_THRESHOLD = 0.5
    TIME_WINDOW = 30
    
    npz_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    if max_videos:
        npz_files = npz_files[:max_videos]
    
    # ========== 第一遍：收集所有max_dist和窗口长度统计 ==========
    print("\n📊 第1/2遍: 扫描所有视频获取全局参考值...")
    max_dists = []
    window_lengths = []
    
    for npz_file in tqdm(npz_files, desc="扫描视频"):
        video_name = npz_file.replace('.npz', '.mp4')
        
        if video_name not in annotations:
            continue
        
        npz_path = os.path.join(NPZ_DIR, npz_file)
        try:
            data = np.load(npz_path)
            detections = data['det']
            features = data['data']
        except:
            continue
        
        accident_frame = annotations[video_name]
        start_frame = max(0, accident_frame - TIME_WINDOW)
        end_frame = min(detections.shape[0], accident_frame + TIME_WINDOW)
        
        # 记录窗口长度
        window_len = end_frame - start_frame
        window_lengths.append(window_len)
        
        # 提取特征窗口
        feats_window = features[start_frame:end_frame, 0, :]
        
        # 计算该视频的最大距离
        if feats_window.shape[0] >= 2:
            distances = np.linalg.norm(feats_window[:-1] - feats_window[1:], axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                max_dists.append(max_dist)
    
    # 计算全局参考值（使用95分位数避免极端值）
    global_max_dist = np.percentile(max_dists, 95) if max_dists else 1.0
    
    print(f"\n📈 窗口长度统计:")
    print(f"   最小: {np.min(window_lengths)}, 最大: {np.max(window_lengths)}, 平均: {np.mean(window_lengths):.1f}")
    print(f"   中位数: {np.median(window_lengths):.0f}, 标准差: {np.std(window_lengths):.1f}")
    print(f"\n🎯 动态变化全局参考值:")
    print(f"   95分位数: {global_max_dist:.6f}")
    print(f"   用于归一化的参考值: {global_max_dist:.6f}")
    
    # ========== 第二遍：用全局参考值重新计算所有指标 ==========
    print(f"\n📊 第2/2遍: 计算所有指标（使用全局归一化）...")
    results = []
    
    for npz_file in tqdm(npz_files, desc="处理视频"):
        video_name = npz_file.replace('.npz', '.mp4')
        
        if video_name not in annotations:
            continue
        
        npz_path = os.path.join(NPZ_DIR, npz_file)
        try:
            data = np.load(npz_path)
            detections = data['det']
            features = data['data']
        except:
            continue
        
        accident_frame = annotations[video_name]
        start_frame = max(0, accident_frame - TIME_WINDOW)
        end_frame = min(detections.shape[0], accident_frame + TIME_WINDOW)
        
        # 提取窗口数据
        dets_window = [detections[i] for i in range(start_frame, end_frame)]
        feats_window = features[start_frame:end_frame, 0, :]
        
        # 置信度过滤
        dets_filtered = []
        for frame_dets in dets_window:
            if frame_dets.size > 0:
                filtered = frame_dets[frame_dets[:, 4] > CONF_THRESHOLD]
                dets_filtered.append(filtered)
            else:
                dets_filtered.append(np.array([]))
        
        # 计算指标（使用全局参考值）
        dynamic = calculate_metrics(feats_window, global_max_dist=global_max_dist)
        complexity = calculate_complexity(dets_filtered)
        has_vru = check_vru_interaction(dets_filtered, VRU_IDS, CAR_IDS)
        
        # 获取真值标签
        label = ground_truth.get(video_name, 0)
        
        results.append({
            'video_name': video_name,
            'accident_frame': accident_frame,
            'dynamic_change': dynamic,
            'scene_complexity': complexity,
            'has_vru_interaction': int(has_vru),
            'human_judgement': label,
            'window_length': end_frame - start_frame
        })
    
    df = pd.DataFrame(results)
    print(f"\n✓ 成功处理 {len(df)} 个视频")
    
    return df, global_max_dist


def analyze_distribution(df):
    """分析指标分布"""
    
    print("\n" + "="*70)
    print("第二阶段: 指标分布分析")
    print("="*70)
    
    # 动态变化分析
    print("\n【动态变化评分 (Dynamic Change)】")
    dyn = df['dynamic_change']
    print(f"  统计值:")
    print(f"    最小值: {dyn.min():.4f}")
    print(f"    最大值: {dyn.max():.4f}")
    print(f"    平均值: {dyn.mean():.4f}")
    print(f"    中位数: {dyn.median():.4f}")
    print(f"  分位数:")
    for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95]:
        val = dyn.quantile(q)
        print(f"    P{int(q*100)}: {val:.4f}")
    
    # 场景复杂度分析
    print("\n【场景复杂度 (Scene Complexity)】")
    cplx = df['scene_complexity']
    print(f"  统计值:")
    print(f"    最小值: {cplx.min()}")
    print(f"    最大值: {cplx.max()}")
    print(f"    平均值: {cplx.mean():.2f}")
    print(f"    中位数: {cplx.median():.2f}")
    print(f"  分位数:")
    for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95]:
        val = cplx.quantile(q)
        print(f"    P{int(q*100)}: {val:.2f}")
    
    # VRU交互分析
    print("\n【VRU交互检测 (Has VRU Interaction)】")
    vru_count = df['has_vru_interaction'].sum()
    print(f"  有VRU交互: {vru_count} ({vru_count/len(df)*100:.1f}%)")
    print(f"  无VRU交互: {len(df)-vru_count} ({(len(df)-vru_count)/len(df)*100:.1f}%)")
    
    # 真值标签分析
    print("\n【人工标注标签 (Human Judgement)】")
    label_counts = df['human_judgement'].value_counts()
    for label in sorted(label_counts.index):
        count = label_counts[label]
        print(f"  Label {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 基于真值的指标分布
    print("\n【按人工标注分层的指标】")
    for label in [0, 1]:
        subset = df[df['human_judgement'] == label]
        if len(subset) == 0:
            continue
        print(f"\n  Label={label} (n={len(subset)}):")
        print(f"    动态变化: {subset['dynamic_change'].mean():.4f} ± {subset['dynamic_change'].std():.4f}")
        print(f"    场景复杂度: {subset['scene_complexity'].mean():.2f} ± {subset['scene_complexity'].std():.2f}")
        print(f"    有VRU: {subset['has_vru_interaction'].sum() / len(subset) * 100:.1f}%")
    
    return df


def suggest_thresholds(df):
    """基于分位数给出阈值建议"""
    
    print("\n" + "="*70)
    print("第三阶段: 基于分位数的阈值建议")
    print("="*70)
    
    dyn = df['dynamic_change']
    cplx = df['scene_complexity']
    
    print("\n【保守策略 (高精度，低召回)】")
    print("  目的: 只筛选高质量样本，容许漏掉部分")
    dynamic_thresh_cons = dyn.quantile(0.8)
    complexity_thresh_cons = cplx.quantile(0.75)
    print(f"    Dynamic >= {dynamic_thresh_cons:.4f} (P80)")
    print(f"    Complexity >= {complexity_thresh_cons:.1f} (P75)")
    
    print("\n【平衡策略 (中等精度，中等召回)】")
    print("  目的: 平衡召回与精度，是推荐方案")
    dynamic_thresh_bal = dyn.quantile(0.6)
    complexity_thresh_bal = cplx.quantile(0.5)
    print(f"    Dynamic >= {dynamic_thresh_bal:.4f} (P60)")
    print(f"    Complexity >= {complexity_thresh_bal:.1f} (P50)")
    
    print("\n【激进策略 (低精度，高召回)】")
    print("  目的: 尽量保留所有潜在样本，接受噪声")
    dynamic_thresh_aggr = dyn.quantile(0.4)
    complexity_thresh_aggr = cplx.quantile(0.25)
    print(f"    Dynamic >= {dynamic_thresh_aggr:.4f} (P40)")
    print(f"    Complexity >= {complexity_thresh_aggr:.1f} (P25)")
    
    suggestions = {
        'conservative': {
            'dynamic_change_threshold': dynamic_thresh_cons,
            'complexity_threshold': complexity_thresh_cons,
        },
        'balanced': {
            'dynamic_change_threshold': dynamic_thresh_bal,
            'complexity_threshold': complexity_thresh_bal,
        },
        'aggressive': {
            'dynamic_change_threshold': dynamic_thresh_aggr,
            'complexity_threshold': complexity_thresh_aggr,
        }
    }
    
    return suggestions


# ============================================================================
# 第三部分: 阈值扫描与性能评估
# ============================================================================

def threshold_sweep(df):
    """阈值扫描，计算性能指标"""
    
    print("\n" + "="*70)
    print("第四阶段: 阈值扫描与性能评估")
    print("="*70)
    
    # 生成扫描范围
    dynamic_thresholds = np.arange(0.0, 1.1, 0.1)
    complexity_thresholds = range(3, 15, 1)
    
    results = []
    
    print(f"\n扫描范围: {len(dynamic_thresholds)} × {len(complexity_thresholds)} = {len(dynamic_thresholds)*len(complexity_thresholds)} 个组合")
    print("计算中...")
    
    for dyn_th in dynamic_thresholds:
        for cplx_th in complexity_thresholds:
            # 应用阈值: 至少满足1个条件
            predicted = ((df['dynamic_change'] >= dyn_th) | (df['scene_complexity'] >= cplx_th)).astype(int)
            
            # 计算指标
            tp = ((predicted == 1) & (df['human_judgement'] == 1)).sum()
            fp = ((predicted == 1) & (df['human_judgement'] == 0)).sum()
            fn = ((predicted == 0) & (df['human_judgement'] == 1)).sum()
            tn = ((predicted == 0) & (df['human_judgement'] == 0)).sum()
            
            # 计算性能
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
            
            results.append({
                'dynamic_threshold': dyn_th,
                'complexity_threshold': cplx_th,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'predicted_positive': tp + fp,
                'actual_positive': tp + fn
            })
    
    results_df = pd.DataFrame(results)
    
    # 找最优阈值
    print("\n【最优阈值 (基于F1)】")
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_row = results_df.loc[best_f1_idx]
    
    print(f"  Dynamic Threshold: {best_f1_row['dynamic_threshold']:.1f}")
    print(f"  Complexity Threshold: {int(best_f1_row['complexity_threshold'])}")
    print(f"  性能指标:")
    print(f"    Recall: {best_f1_row['recall']:.4f}")
    print(f"    Precision: {best_f1_row['precision']:.4f}")
    print(f"    F1: {best_f1_row['f1']:.4f}")
    print(f"    TP/FP/FN: {int(best_f1_row['tp'])}/{int(best_f1_row['fp'])}/{int(best_f1_row['fn'])}")
    
    # 显示Top-10
    print("\n【Top-10 最优阈值组合 (按F1)】")
    top_k = results_df.nlargest(10, 'f1')[['dynamic_threshold', 'complexity_threshold', 'recall', 'precision', 'f1']]
    for idx, row in top_k.iterrows():
        print(f"  Dyn={row['dynamic_threshold']:.1f}, Cplx={int(row['complexity_threshold'])}: "
              f"R={row['recall']:.3f}, P={row['precision']:.3f}, F1={row['f1']:.3f}")
    
    return results_df, best_f1_row


# ============================================================================
# 第四部分: 结果对比与导出
# ============================================================================

def compare_thresholds(df, current_config, new_config):
    """对比新旧阈值的筛选结果"""
    
    print("\n" + "="*70)
    print("第五阶段: 新旧阈值对比")
    print("="*70)
    
    # 应用旧阈值
    old_pred = ((df['dynamic_change'] >= current_config['dynamic']) | 
                (df['scene_complexity'] >= current_config['complexity'])).astype(int)
    
    # 应用新阈值
    new_pred = ((df['dynamic_change'] >= new_config['dynamic']) | 
                (df['scene_complexity'] >= new_config['complexity'])).astype(int)
    
    # 统计
    old_pass = (old_pred == 1).sum()
    new_pass = (new_pred == 1).sum()
    old_correct = ((old_pred == 1) & (df['human_judgement'] == 1)).sum()
    new_correct = ((new_pred == 1) & (df['human_judgement'] == 1)).sum()
    
    print(f"\n【当前阈值】")
    print(f"  Dynamic >= {current_config['dynamic']:.1f}, Complexity >= {current_config['complexity']}")
    print(f"  筛选数: {old_pass} ({old_pass/len(df)*100:.1f}%)")
    print(f"  人工通过的: {old_correct}/{old_pass if old_pass > 0 else 1} "
          f"({old_correct/old_pass*100 if old_pass > 0 else 0:.1f}%)")
    
    print(f"\n【建议新阈值】")
    print(f"  Dynamic >= {new_config['dynamic']:.1f}, Complexity >= {new_config['complexity']:.0f}")
    print(f"  筛选数: {new_pass} ({new_pass/len(df)*100:.1f}%)")
    print(f"  人工通过的: {new_correct}/{new_pass if new_pass > 0 else 1} "
          f"({new_correct/new_pass*100 if new_pass > 0 else 0:.1f}%)")
    
    print(f"\n【变化分析】")
    print(f"  筛选数变化: {new_pass - old_pass:+d} ({(new_pass-old_pass)/old_pass*100:+.1f}%)")
    print(f"  精度变化: {new_correct/new_pass if new_pass > 0 else 0:.1f}% "
          f"(vs {old_correct/old_pass if old_pass > 0 else 0:.1f}%)")
    
    # 细节分析
    improved = ((old_pred == 0) & (new_pred == 1) & (df['human_judgement'] == 1))
    worsened = ((old_pred == 1) & (new_pred == 0) & (df['human_judgement'] == 1))
    extra_fp = ((old_pred == 0) & (new_pred == 1) & (df['human_judgement'] == 0))
    
    print(f"\n【细节分析】")
    print(f"  新增发现(True Positive): {improved.sum()}")
    print(f"  误删除(False Negative): {worsened.sum()}")
    print(f"  新增误报(False Positive): {extra_fp.sum()}")
    
    return {
        'old': {
            'predictions': old_pred,
            'pass_count': old_pass,
            'correct_count': old_correct
        },
        'new': {
            'predictions': new_pred,
            'pass_count': new_pass,
            'correct_count': new_correct
        }
    }


def export_results(df, results_df, best_config, comparison, output_dir):
    """导出分析结果"""
    
    print("\n" + "="*70)
    print("第六阶段: 导出结果")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出分布分析
    dist_report = {
        'dynamic_change': {
            'min': float(df['dynamic_change'].min()),
            'max': float(df['dynamic_change'].max()),
            'mean': float(df['dynamic_change'].mean()),
            'median': float(df['dynamic_change'].median()),
            'std': float(df['dynamic_change'].std()),
            'quantiles': {
                f'p{int(q*100)}': float(df['dynamic_change'].quantile(q))
                for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95]
            }
        },
        'scene_complexity': {
            'min': int(df['scene_complexity'].min()),
            'max': int(df['scene_complexity'].max()),
            'mean': float(df['scene_complexity'].mean()),
            'median': float(df['scene_complexity'].median()),
            'std': float(df['scene_complexity'].std()),
            'quantiles': {
                f'p{int(q*100)}': float(df['scene_complexity'].quantile(q))
                for q in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95]
            }
        }
    }
    
    with open(os.path.join(output_dir, '01_distribution_analysis.json'), 'w') as f:
        json.dump(dist_report, f, indent=2)
    print(f"✓ 已导出: 01_distribution_analysis.json")
    
    # 导出阈值扫描结果
    results_df.to_csv(os.path.join(output_dir, '02_threshold_sweep_results.csv'), index=False)
    print(f"✓ 已导出: 02_threshold_sweep_results.csv ({len(results_df)} 行)")
    
    # 导出最优阈值配置
    best_config_dict = {
        'dynamic_change_threshold': float(best_config['dynamic_threshold']),
        'scene_complexity_threshold': float(best_config['complexity_threshold']),
        'recall': float(best_config['recall']),
        'precision': float(best_config['precision']),
        'f1': float(best_config['f1']),
        'true_positive': int(best_config['tp']),
        'false_positive': int(best_config['fp']),
        'false_negative': int(best_config['fn'])
    }
    
    with open(os.path.join(output_dir, '03_optimal_config.json'), 'w') as f:
        json.dump(best_config_dict, f, indent=2)
    print(f"✓ 已导出: 03_optimal_config.json")
    
    # 导出新阈值的筛选结果
    new_pred_videos = df[comparison['new']['predictions'] == 1].copy()
    new_pred_videos.to_csv(os.path.join(output_dir, '04_new_threshold_filtered_videos.csv'), index=False)
    print(f"✓ 已导出: 04_new_threshold_filtered_videos.csv ({len(new_pred_videos)} 条)")
    
    # 导出对比报告
    with open(os.path.join(output_dir, '05_comparison_report.txt'), 'w') as f:
        f.write("筛选阈值优化报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("【当前配置】\n")
        f.write(f"Dynamic >= 1.0, Complexity >= 6\n")
        f.write(f"筛选数: {comparison['old']['pass_count']}\n")
        f.write(f"准确率: {comparison['old']['correct_count']}/{comparison['old']['pass_count']} "
                f"({comparison['old']['correct_count']/comparison['old']['pass_count']*100:.1f}%)\n\n")
        
        f.write("【最优配置 (基于F1)】\n")
        f.write(f"Dynamic >= {best_config['dynamic_threshold']:.1f}, "
                f"Complexity >= {int(best_config['complexity_threshold'])}\n")
        f.write(f"筛选数: {comparison['new']['pass_count']}\n")
        f.write(f"准确率: {comparison['new']['correct_count']}/{comparison['new']['pass_count']} "
                f"({comparison['new']['correct_count']/comparison['new']['pass_count']*100:.1f}%)\n")
        f.write(f"Recall: {best_config['recall']:.4f}\n")
        f.write(f"Precision: {best_config['precision']:.4f}\n")
        f.write(f"F1: {best_config['f1']:.4f}\n\n")
        
        f.write("【性能对比】\n")
        f.write(f"筛选数变化: {comparison['new']['pass_count'] - comparison['old']['pass_count']:+d}\n")
        f.write(f"精度变化: {comparison['new']['correct_count']/comparison['new']['pass_count']*100:.1f}% "
                f"(vs {comparison['old']['correct_count']/comparison['old']['pass_count']*100:.1f}%)\n")
    
    print(f"✓ 已导出: 05_comparison_report.txt")
    
    print(f"\n✓ 所有结果已保存至: {output_dir}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  筛选阈值分析与优化（全局归一化）".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Step 1: 计算全量指标（返回df和全局参考值）
    result = compute_all_metrics()
    if result is None or result[0] is None:
        return
    df, global_max_dist = result
    
    # Step 2: 分析分布
    analyze_distribution(df)
    
    # Step 3: 给出建议
    suggestions = suggest_thresholds(df)
    
    # Step 4: 阈值扫描
    results_df, best_f1_row = threshold_sweep(df)
    
    # Step 5: 对比
    current_config = {'dynamic': 1.0, 'complexity': 6}
    new_config = {
        'dynamic': best_f1_row['dynamic_threshold'],
        'complexity': best_f1_row['complexity_threshold']
    }
    
    comparison = compare_thresholds(df, current_config, new_config)
    
    # Step 6: 导出
    export_results(df, results_df, best_f1_row, comparison, ANALYSIS_OUTPUT)
    
    print("\n" + "="*70)
    print("✓ 分析完成！")
    print("="*70)


if __name__ == '__main__':
    main()
