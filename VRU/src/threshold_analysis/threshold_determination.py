#!/usr/bin/env python3
"""
基于统计分布的阈值自动确定方案

背景：不同的统计方法会导致不同的筛选结果，本脚本提供多种
"科学合理"的阈值确定方法，可根据实际工作量需求选择。

方法概览：
1. Elbow法：找分布的自然断点
2. 标准差法：mean + n*std（n为自由度）
3. 分位数法：P95/P90/P75等关键点
4. 多指标加权：综合Complexity+Dynamic
5. 峰度分析：基于分布的"肥尾"特性
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

ROOT_DIR = "/home/24068286g/UString"
CCD_ROOT = os.path.join(ROOT_DIR, 'data', 'crash')
NPZ_DIR = os.path.join(CCD_ROOT, 'yolo_features', 'positive')
ANNOTATION_FILE = os.path.join(CCD_ROOT, 'videos', 'Crash-1500.txt')

# ============================================================================
# 工具函数：加载数据
# ============================================================================

def load_annotations(file_path):
    """加载事故帧标注"""
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
        pass
    return annotations


def compute_metrics(time_window=30, conf_threshold=0.5):
    """计算所有视频的指标"""
    
    annotations = load_annotations(ANNOTATION_FILE)
    if not annotations:
        print("✗ 无法加载标注")
        return None
    
    VRU_IDS = {0, 1, 3}
    CAR_IDS = {2, 5, 7}
    
    npz_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    
    # 第一遍：获取全局参考值
    print("📊 收集全局参考值...")
    max_dists = []
    
    for npz_file in tqdm(npz_files, desc="第1/2遍"):
        video_name = npz_file.replace('.npz', '.mp4')
        if video_name not in annotations:
            continue
        
        npz_path = os.path.join(NPZ_DIR, npz_file)
        try:
            data = np.load(npz_path)
            features = data['data']
        except:
            continue
        
        accident_frame = annotations[video_name]
        start_frame = max(0, accident_frame - time_window)
        end_frame = min(features.shape[0], accident_frame + time_window)
        feats_window = features[start_frame:end_frame, 0, :]
        
        if feats_window.shape[0] >= 2:
            distances = np.linalg.norm(feats_window[:-1] - feats_window[1:], axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                max_dists.append(max_dist)
    
    global_ref = np.percentile(max_dists, 95)
    print(f"✓ 全局参考值 (P95): {global_ref:.6f}\n")
    
    # 第二遍：计算指标
    print("📊 计算所有指标...")
    results = []
    
    for npz_file in tqdm(npz_files, desc="第2/2遍"):
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
        start_frame = max(0, accident_frame - time_window)
        end_frame = min(features.shape[0], accident_frame + time_window)
        
        dets_window = [detections[i] for i in range(start_frame, end_frame)]
        feats_window = features[start_frame:end_frame, 0, :]
        
        # Dynamic Change
        if feats_window.shape[0] >= 2:
            distances = np.linalg.norm(feats_window[:-1] - feats_window[1:], axis=1)
            max_dist = np.max(distances) if np.max(distances) > 0 else 1e-6
            dynamic = float(np.max(distances / global_ref))
        else:
            dynamic = 0.0
        
        # Scene Complexity
        max_objs = 0
        for frame_dets in dets_window:
            if frame_dets.size > 0:
                filtered = frame_dets[frame_dets[:, 4] > conf_threshold]
                max_objs = max(max_objs, filtered.shape[0])
        
        # VRU Interaction
        has_vru = False
        for frame_dets in dets_window:
            if frame_dets.size > 0:
                filtered = frame_dets[frame_dets[:, 4] > conf_threshold]
                classes = set(int(obj[5]) for obj in filtered)
                if (classes & VRU_IDS) and (classes & CAR_IDS):
                    has_vru = True
                    break
        
        results.append({
            'video_name': video_name,
            'dynamic_change': dynamic,
            'scene_complexity': max_objs,
            'has_vru': int(has_vru)
        })
    
    df = pd.DataFrame(results)
    return df


# ============================================================================
# 阈值确定方法
# ============================================================================

class ThresholdDeterminer:
    """多种阈值确定方法的集合"""
    
    def __init__(self, df):
        self.df = df
        self.dyn = df['dynamic_change'].values
        self.cplx = df['scene_complexity'].values
        self.total = len(df)
    
    def method_1_stddev(self, metric='complexity', n_std=1.0, direction='above'):
        """
        标准差法：mean ± n*std
        
        Args:
            metric: 'complexity' 或 'dynamic'
            n_std: 标准差倍数（1.0 = mean+1*std，更严格）
            direction: 'above' (mean+n*std) 或 'below' (mean-n*std)
        """
        data = self.cplx if metric == 'complexity' else self.dyn
        mean = data.mean()
        std = data.std()
        
        threshold = mean + n_std * std if direction == 'above' else mean - n_std * std
        count = (data >= threshold).sum() if direction == 'above' else (data <= threshold).sum()
        pct = count / self.total * 100
        
        return {
            'method': f'标准差法 ({metric}) mean {direction} {n_std}*std',
            'threshold': float(threshold),
            'selected_count': int(count),
            'percentage': float(pct),
            'mean': float(mean),
            'std': float(std),
        }
    
    def method_2_percentile(self, metric='complexity', percentile=75):
        """
        分位数法：取上位分位数
        
        实质：只选择最好的前(100-percentile)%的样本
        """
        data = self.cplx if metric == 'complexity' else self.dyn
        threshold = np.percentile(data, percentile)
        count = (data >= threshold).sum()
        pct = count / self.total * 100
        
        return {
            'method': f'分位数法 ({metric}) P{percentile}',
            'threshold': float(threshold),
            'selected_count': int(count),
            'percentage': float(pct),
            'description': f'选择前{100-percentile:.0f}%的最高质量样本',
        }
    
    def method_3_elbow(self, metric='complexity'):
        """
        Elbow法：寻找分布的自然断点（二阶导数最大）
        """
        data = self.cplx if metric == 'complexity' else self.dyn
        
        # 排序并计算差分
        sorted_data = np.sort(data)[::-1]  # 从大到小
        diff1 = np.diff(sorted_data)
        diff2 = np.diff(diff1)  # 二阶导数
        
        # 找最大曲率点（二阶导数最大）
        elbow_idx = np.argmax(np.abs(diff2)) + 1
        threshold = sorted_data[elbow_idx]
        count = (data >= threshold).sum()
        pct = count / self.total * 100
        
        return {
            'method': f'Elbow法 ({metric})',
            'threshold': float(threshold),
            'selected_count': int(count),
            'percentage': float(pct),
            'elbow_position': int(elbow_idx),
            'description': '寻找分布的自然断点',
        }
    
    def method_4_combined_weighted(self, cplx_weight=0.7, dyn_weight=0.3):
        """
        加权综合法：结合Complexity和Dynamic
        
        综合评分 = cplx_normalized * cplx_weight + dyn_normalized * dyn_weight
        选择评分在上分位数的样本
        """
        # 归一化
        cplx_norm = (self.cplx - self.cplx.min()) / (self.cplx.max() - self.cplx.min() + 1e-6)
        dyn_norm = (self.dyn - self.dyn.min()) / (self.dyn.max() - self.dyn.min() + 1e-6)
        
        # 加权评分
        scores = cplx_norm * cplx_weight + dyn_norm * dyn_weight
        
        # 选择前25%最好的
        threshold = np.percentile(scores, 75)
        selected = scores >= threshold
        count = selected.sum()
        pct = count / self.total * 100
        
        return {
            'method': f'加权综合法 (Complexity {cplx_weight}, Dynamic {dyn_weight})',
            'threshold': float(threshold),
            'selected_count': int(count),
            'percentage': float(pct),
            'selected_videos': self.df[selected]['video_name'].tolist(),
        }
    
    def method_5_distribution_shape(self, metric='complexity'):
        """
        基于分布形状的自适应法
        
        使用峰度(kurtosis)和偏度(skewness)来确定最优切分点
        """
        data = self.cplx if metric == 'complexity' else self.dyn
        
        mean = data.mean()
        std = data.std()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # 如果分布是正态的（skew~0, kurt~0），用标准差
        # 如果右偏（skew>0），应该用更高的阈值来避免极端值影响
        if abs(skewness) < 0.5:
            # 近似正态 → mean + 1*std
            threshold = mean + 1.0 * std
        else:
            # 非正态 → 用分位数
            threshold = np.percentile(data, 75)
        
        count = (data >= threshold).sum()
        pct = count / self.total * 100
        
        return {
            'method': f'分布自适应法 ({metric})',
            'threshold': float(threshold),
            'selected_count': int(count),
            'percentage': float(pct),
            'distribution_analysis': {
                'mean': float(mean),
                'std': float(std),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'shape': '近似正态' if abs(skewness) < 0.5 else '非正态分布',
            }
        }
    
    def generate_report(self, target_reduction_rate=0.3):
        """
        生成完整的阈值建议报告
        
        Args:
            target_reduction_rate: 目标削减率（如0.3 = 减少30%）
        """
        
        print("\n" + "="*80)
        print("【多方法阈值确定报告】")
        print("="*80)
        print(f"总视频数: {self.total}")
        print(f"目标削减率: {target_reduction_rate*100:.0f}% (保留 {int(self.total*(1-target_reduction_rate))} 个)")
        
        results = {}
        
        print("\n" + "-"*80)
        print("方法1: 标准差法（推荐用于近似正态分布）")
        print("-"*80)
        
        for n_std in [0.5, 1.0, 1.5, 2.0]:
            result = self.method_1_stddev('complexity', n_std)
            results[f'stddev_cplx_{n_std}'] = result
            print(f"  mean + {n_std}*std = {result['threshold']:.2f}")
            print(f"    → 选择 {result['selected_count']} 个 ({result['percentage']:.1f}%)")
        
        print("\n" + "-"*80)
        print("方法2: 分位数法（目标保留某个百分比）")
        print("-"*80)
        
        for percentile in [50, 60, 70, 75, 80, 90]:
            result = self.method_2_percentile('complexity', percentile)
            results[f'percentile_cplx_{percentile}'] = result
            print(f"  P{percentile}:")
            print(f"    阈值 = {result['threshold']:.2f}")
            print(f"    → 选择 {result['selected_count']} 个 ({result['percentage']:.1f}%)")
        
        print("\n" + "-"*80)
        print("方法3: Elbow法（找自然断点）")
        print("-"*80)
        
        result = self.method_3_elbow('complexity')
        results['elbow_cplx'] = result
        print(f"  自然断点:")
        print(f"    阈值 = {result['threshold']:.2f}")
        print(f"    → 选择 {result['selected_count']} 个 ({result['percentage']:.1f}%)")
        
        print("\n" + "-"*80)
        print("方法4: 加权综合法")
        print("-"*80)
        
        result = self.method_4_combined_weighted(0.7, 0.3)
        results['weighted_combined'] = result
        print(f"  Complexity 70% + Dynamic 30%:")
        print(f"    → 选择 {result['selected_count']} 个 ({result['percentage']:.1f}%)")
        
        print("\n" + "-"*80)
        print("方法5: 分布自适应法")
        print("-"*80)
        
        result = self.method_5_distribution_shape('complexity')
        results['adaptive_dist'] = result
        dist_info = result['distribution_analysis']
        print(f"  分布特征: {dist_info['shape']}")
        print(f"    Mean: {dist_info['mean']:.2f}, Std: {dist_info['std']:.2f}")
        print(f"    Skewness: {dist_info['skewness']:.3f}, Kurtosis: {dist_info['kurtosis']:.3f}")
        print(f"    阈值 = {result['threshold']:.2f}")
        print(f"    → 选择 {result['selected_count']} 个 ({result['percentage']:.1f}%)")
        
        print("\n" + "="*80)
        print("【建议选择】")
        print("="*80)
        print("""
        根据应用场景选择：
        
        ✅ 场景1: 想要"严格"的科学方法 
           → 使用 Elbow法 或 分布自适应法
           特点：依据分布的自然特征，看起来最"客观"
        
        ✅ 场景2: 想要特定的保留比例（如保留75%）
           → 使用 分位数法 P75
           特点：明确说出保留比例，目标明确
        
        ✅ 场景3: 想要平衡多个指标
           → 使用 加权综合法
           特点：结合Complexity和Dynamic，显得更全面
        
        ✅ 场景4: 数据接近正态分布
           → 使用 标准差法 (mean + 1.0*std)
           特点：统计学教科书级的方法，最严谨
        """)
        
        print("="*80)
        print("✓ 报告生成完成")
        print("="*80)
        
        return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  基于统计分布的阈值自动确定工具".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # 加载数据
    df = compute_metrics()
    if df is None:
        return
    
    print(f"\n✓ 成功加载 {len(df)} 个视频的指标\n")
    
    # 创建阈值确定器
    determiner = ThresholdDeterminer(df)
    
    # 生成报告
    results = determiner.generate_report(target_reduction_rate=0.3)
    
    # 保存结果
    output_file = '/home/24068286g/UString/VRU/threshold_analysis/threshold_methods_comparison.json'
    with open(output_file, 'w') as f:
        # 转换为可序列化的格式
        serializable = {}
        for key, val in results.items():
            if isinstance(val, dict):
                serializable[key] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in val.items()
                }
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ 结果已保存至: {output_file}")


if __name__ == '__main__':
    main()
