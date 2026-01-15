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
from tqdm import tqdm

# ============================================================================
# 配置与路径
# ============================================================================

ROOT_DIR = "/home/24068286g/UString"
CCD_ROOT = os.path.join(ROOT_DIR, 'data', 'crash')
VRU_ROOT = "/home/24068286g/CCD_VQA/VRU"

ANNOTATION_FILE = os.path.join(CCD_ROOT, 'videos', 'Crash-1500.txt')
NPZ_DIR = os.path.join(CCD_ROOT, 'yolo_features', 'positive')
OUTPUT_DIR = os.path.join(VRU_ROOT, 'output')
ANALYSIS_OUTPUT = os.path.join(VRU_ROOT, 'src', 'threshold_analysis', 'threshold_analysis')

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


# ============================================================================
# 第二部分: 全量数据分析
# ============================================================================

def compute_all_metrics(max_videos=None):
    """计算所有视频的指标分布（两阶段：先计算全局参考值，再归一化）"""
    
    print("\n" + "="*70)
    print("第一阶段: 计算全量视频指标分布（全局归一化）")
    print("="*70)
    
    annotations = load_annotations(ANNOTATION_FILE)
    
    if not annotations:
        print("✗ 无法加载标注")
        return None, None
    
    # 配置
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
        
        # 帧内：对高置信度检测取平均特征（与下游 pipeline 一致）
        frame_avg_features = []
        for t in range(start_frame, end_frame):
            frame_dets = detections[t]
            if frame_dets.size > 0:
                high_conf_mask = frame_dets[:, 4] > CONF_THRESHOLD
                high_conf_indices = np.where(high_conf_mask)[0]
                if len(high_conf_indices) > 0:
                    frame_feat = np.mean(features[t, high_conf_indices, :], axis=0)
                else:
                    frame_feat = features[t, 0, :]
            else:
                frame_feat = features[t, 0, :]
            frame_avg_features.append(frame_feat)
        feats_window = np.array(frame_avg_features) if len(frame_avg_features) > 0 else np.array([])
        
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
        
        dets_window = [detections[i] for i in range(start_frame, end_frame)]
        
        # 帧内平均特征（与全局参考一致）
        frame_avg_features = []
        for t, frame_dets in enumerate(dets_window):
            idx = start_frame + t
            if frame_dets.size > 0:
                high_conf_mask = frame_dets[:, 4] > CONF_THRESHOLD
                high_conf_indices = np.where(high_conf_mask)[0]
                if len(high_conf_indices) > 0:
                    frame_feat = np.mean(features[idx, high_conf_indices, :], axis=0)
                else:
                    frame_feat = features[idx, 0, :]
            else:
                frame_feat = features[idx, 0, :]
            frame_avg_features.append(frame_feat)
        feats_window = np.array(frame_avg_features) if len(frame_avg_features) > 0 else np.array([])
        
        # 置信度过滤用于复杂度
        dets_filtered = []
        for frame_dets in dets_window:
            if frame_dets.size > 0:
                filtered = frame_dets[frame_dets[:, 4] > CONF_THRESHOLD]
                dets_filtered.append(filtered)
            else:
                dets_filtered.append(np.array([]))
        
        dynamic = calculate_metrics(feats_window, global_max_dist=global_max_dist)
        complexity = calculate_complexity(dets_filtered)
        
        results.append({
            'video_name': video_name,
            'accident_frame': accident_frame,
            'dynamic_change': dynamic,
            'scene_complexity': complexity,
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
    for q in [0.25, 0.5,0.6, 0.7, 0.8, 0.9, 0.95]:
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
    for q in [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        val = cplx.quantile(q)
        print(f"    P{int(q*100)}: {val:.2f}")
    
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


def export_basic_reports(df, suggestions):
    """导出基础分布与分位数建议，便于 pipeline 复用"""
    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)

    dist_report = {
        'dynamic_change': {
            'min': float(df['dynamic_change'].min()),
            'max': float(df['dynamic_change'].max()),
            'mean': float(df['dynamic_change'].mean()),
            'median': float(df['dynamic_change'].median()),
            'std': float(df['dynamic_change'].std()),
            'quantiles': {
                f'p{int(q*100)}': float(df['dynamic_change'].quantile(q))
                for q in [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
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
                for q in [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            }
        }
    }

    with open(os.path.join(ANALYSIS_OUTPUT, '01_distribution_analysis.json'), 'w') as f:
        json.dump(dist_report, f, indent=2)

    with open(os.path.join(ANALYSIS_OUTPUT, '02_threshold_suggestions.json'), 'w') as f:
        json.dump(suggestions, f, indent=2)

    df.to_csv(os.path.join(ANALYSIS_OUTPUT, '00_raw_metrics.csv'), index=False)


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  筛选阈值分析（Dynamic + Complexity）".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    result = compute_all_metrics()
    if result is None or result[0] is None:
        return
    df, _ = result

    analyze_distribution(df)
    suggestions = suggest_thresholds(df)
    export_basic_reports(df, suggestions)

    print("\n" + "="*70)
    print("✓ 分析完成，结果已导出至 threshold_analysis 目录")
    print("="*70)


if __name__ == '__main__':
    main()
