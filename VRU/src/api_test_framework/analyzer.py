"""
分析模块
负责相似度计算、可视化和结果分析
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAnalyzer:
    """相似度计算与分析"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.cosine_similarity = cosine_similarity

    def compute_similarity(self, responses_dict: Dict[str, str]) -> Dict[str, float]:
        """
        计算多个响应之间的 cosine 相似度。
        返回：{"model_a_model_b": 0.85, ...}
        """
        model_names = list(responses_dict.keys())
        texts = [responses_dict[m] for m in model_names]

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            sim_matrix = self.cosine_similarity(tfidf_matrix)
        except:
            return {}

        similarities = {}
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i < j:
                    key = f"{m1}_vs_{m2}"
                    similarities[key] = float(sim_matrix[i][j])

        return similarities

    def rank_similarities(self, similarities: Dict[str, float]) -> List[Tuple[str, float]]:
        """按相似度排序"""
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    def plot_heatmap(self, responses_dict: Dict[str, str], video_id: str, save_path: str = None):
        """绘制相似度热力图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("⚠️ matplotlib/seaborn 未安装，跳过热力图")
            return

        model_names = list(responses_dict.keys())
        texts = [responses_dict[m] for m in model_names]

        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except:
            print(f"⚠️ 相似度计算失败: {video_id}")
            return

        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=model_names, yticklabels=model_names,
                    cbar_kws={"label": "Cosine Similarity"})
        plt.title(f"相似度热力图 - {video_id}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"💾 热力图保存: {save_path}")
        plt.show()


class ResultVisualizer:
    """结果可视化与分析"""

    def __init__(self, results_json_path: str):
        with open(results_json_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def print_summary(self):
        """打印摘要统计"""
        print("📊 结果摘要")
        print("=" * 80)
        print(f"任务: {self.results['task']}")
        print(f"时间戳: {self.results['timestamp']}")
        print(f"总视频: {self.results['total_videos']}")
        print(f"使用模型: {', '.join(self.results['models_used'])}")

    def print_video_details(self, video_idx: int):
        """打印单个视频的详细结果"""
        video = self.results['videos'][video_idx]
        print(f"\n🎬 {video['video_id']}")
        print("-" * 80)
        print(f"帧数: {video['frames_count']}")

        for model in self.results['models_used']:
            response = video['responses'].get(model, "")
            error = video['errors'].get(model)
            latency = video['latencies_seconds'].get(model, 0)

            if error:
                print(f"\n❌ {model} (错误):")
                print(f"   {error}")
            else:
                print(f"\n✓ {model} ({latency:.2f}s):")
                print(f"   {response[:200]}...")

        if video['similarity_ranking']:
            print(f"\n相似度排名（Top 10）:")
            for rank, (pair, sim) in enumerate(video['similarity_ranking'][:10], 1):
                print(f"  {rank:2d}. {pair:40s}: {sim:.4f}")

    def export_rankings_to_csv(self, output_path: str = None):
        """导出所有视频的相似度排名为 CSV"""
        if output_path is None:
            output_path = Path("similarity_rankings.csv")

        rows = []
        for video in self.results['videos']:
            for rank, (pair, sim) in enumerate(video['similarity_ranking'], 1):
                rows.append({
                    'video_id': video['video_id'],
                    'rank': rank,
                    'model_pair': pair,
                    'similarity': sim
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✓ 排名已导出: {output_path}")
        return df

    def plot_model_agreement(self, video_idx: int = None):
        """绘制模型一致性统计"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib 未安装，跳过绘图")
            return

        if video_idx is not None:
            video = self.results['videos'][video_idx]
            similarities = [sim for _, sim in video['similarity_ranking']]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(similarities, bins=10, color='skyblue', edgecolor='black')
            axes[0].set_xlabel("Cosine Similarity")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(f"相似度分布 - {video['video_id']}")
            axes[0].grid(alpha=0.3)

            axes[1].axis('off')
            stats_text = f"""
            视频: {video['video_id']}

            相似度统计:
            • 最小: {min(similarities):.4f}
            • 最大: {max(similarities):.4f}
            • 平均: {np.mean(similarities):.4f}
            • 中位数: {np.median(similarities):.4f}
            • 标准差: {np.std(similarities):.4f}

            模型对数: {len(similarities)}
            """
            axes[1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace')

            plt.tight_layout()
            plt.show()
        else:
            all_similarities = []
            video_names = []

            for video in self.results['videos']:
                sims = [sim for _, sim in video['similarity_ranking']]
                if sims:
                    all_similarities.append(np.mean(sims))
                    video_names.append(video['video_id'][:15])

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(range(len(all_similarities)), all_similarities, color='lightgreen', edgecolor='black')
            ax.set_xticks(range(len(video_names)))
            ax.set_xticklabels(video_names, rotation=45, ha='right')
            ax.set_ylabel("平均相似度")
            ax.set_title("模型间平均一致性（跨视频）")
            ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()


class ResultAnalyzer:
    """深度结果分析"""

    def __init__(self, results_json_path: str):
        with open(results_json_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def quality_check(self):
        """数据质量检查"""
        print("🔍 数据质量检查")
        print("=" * 80)

        models = self.results['models_used']
        videos = self.results['videos']

        print(f"📊 总体指标:")
        print(f"  • 视频数: {len(videos)}")
        print(f"  • 模型数: {len(models)}")
        print(f"  • 总推理数: {len(videos) * len(models)}")

        print(f"\n✓ 模型成功率:")
        for model in models:
            successes = sum(1 for v in videos if v['responses'].get(model))
            rate = 100 * successes / len(videos)
            status = "✓" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"  {status} {model:25s}: {successes:3d}/{len(videos):3d} ({rate:5.1f}%)")

        print(f"\n📝 响应长度（字符数）:")
        for model in models:
            lengths = [len(v['responses'].get(model, '')) for v in videos
                      if v['responses'].get(model)]
            if lengths:
                print(f"  {model:25s}: "
                      f"min={min(lengths):5d}, max={max(lengths):5d}, "
                      f"avg={int(np.mean(lengths)):5d}")

        print(f"\n⏱️ 推理延迟（秒）:")
        for model in models:
            latencies = [v['latencies_seconds'].get(model, 0) for v in videos
                        if v['latencies_seconds'].get(model, 0) > 0]
            if latencies:
                print(f"  {model:25s}: "
                      f"min={min(latencies):6.2f}, max={max(latencies):6.2f}, "
                      f"avg={np.mean(latencies):6.2f}")

        print(f"\n📈 模型间相似度（TF-IDF cosine）:")
        all_sims = []
        for video in videos:
            all_sims.extend([sim for _, sim in video['similarity_ranking']])

        if all_sims:
            print(f"  • 最小: {min(all_sims):.4f}")
            print(f"  • 最大: {max(all_sims):.4f}")
            print(f"  • 平均: {np.mean(all_sims):.4f}")
            print(f"  • 中位数: {np.median(all_sims):.4f}")
            print(f"  • 标准差: {np.std(all_sims):.4f}")

    def identify_best_agreement_videos(self, top_n: int = 5):
        """找出模型一致性最高的视频"""
        videos_with_avg_sim = []

        for video in self.results['videos']:
            sims = [sim for _, sim in video['similarity_ranking']]
            if sims:
                avg_sim = np.mean(sims)
                videos_with_avg_sim.append((video['video_id'], avg_sim))

        videos_with_avg_sim.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🎯 模型一致性最高的 {min(top_n, len(videos_with_avg_sim))} 个视频:")
        for rank, (video_id, avg_sim) in enumerate(videos_with_avg_sim[:top_n], 1):
            print(f"  {rank}. {video_id:30s}: {avg_sim:.4f}")

    def identify_difficult_videos(self, top_n: int = 5):
        """找出模型差异最大的视频"""
        videos_with_std = []

        for video in self.results['videos']:
            sims = [sim for _, sim in video['similarity_ranking']]
            if sims:
                std_sim = np.std(sims)
                videos_with_std.append((video['video_id'], std_sim, np.mean(sims)))

        videos_with_std.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🔥 模型差异最大的 {min(top_n, len(videos_with_std))} 个视频:")
        for rank, (video_id, std, avg) in enumerate(videos_with_std[:top_n], 1):
            print(f"  {rank}. {video_id:30s}: σ={std:.4f}, μ={avg:.4f}")

    def export_full_report(self, output_path: str = None):
        """导出完整分析报告（Markdown）"""
        if output_path is None:
            output_path = Path("analysis_report.md")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 多模型评估分析报告\n\n")
            f.write(f"## 基本信息\n\n")
            f.write(f"- **任务**: {self.results['task']}\n")
            f.write(f"- **时间**: {self.results['timestamp']}\n")
            f.write(f"- **总视频**: {len(self.results['videos'])}\n")
            f.write(f"- **使用模型**: {', '.join(self.results['models_used'])}\n\n")

            f.write(f"## 模型性能\n\n")

            for model in self.results['models_used']:
                f.write(f"### {model}\n\n")

                successes = sum(1 for v in self.results['videos']
                               if v['responses'].get(model))
                rate = 100 * successes / len(self.results['videos'])
                f.write(f"- **成功率**: {rate:.1f}% ({successes}/{len(self.results['videos'])})\n")

                latencies = [v['latencies_seconds'].get(model, 0)
                            for v in self.results['videos']
                            if v['latencies_seconds'].get(model, 0) > 0]
                if latencies:
                    f.write(f"- **平均延迟**: {np.mean(latencies):.2f}s\n")

                lengths = [len(v['responses'].get(model, ''))
                          for v in self.results['videos']
                          if v['responses'].get(model)]
                if lengths:
                    f.write(f"- **平均响应长度**: {int(np.mean(lengths))} 字符\n")

                f.write("\n")

            f.write(f"## 相似度分析\n\n")
            all_sims = []
            for video in self.results['videos']:
                all_sims.extend([sim for _, sim in video['similarity_ranking']])

            if all_sims:
                f.write(f"- **最小相似度**: {min(all_sims):.4f}\n")
                f.write(f"- **最大相似度**: {max(all_sims):.4f}\n")
                f.write(f"- **平均相似度**: {np.mean(all_sims):.4f}\n")
                f.write(f"- **中位数**: {np.median(all_sims):.4f}\n")
                f.write(f"- **标准差**: {np.std(all_sims):.4f}\n\n")

        print(f"✓ 报告已导出: {output_path}")
