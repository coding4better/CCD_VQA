#!/usr/bin/env python3
"""
Unified benchmark result analysis with visualization.

Features:
- Parse all result JSON files (including nested folders like phase2/)
- Compute raw accuracy, BSS, chance-corrected skill score
- Compute per-question accuracy (Q1-Q6)
- Compute clustered statistics for Q1-3 and Q4-6
- Export CSV tables and PNG visualizations

Usage:
    python analyze_result_suite.py
    python analyze_result_suite.py --result-dir result --output-dir analysis_suite
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_MAX_QUESTIONS = 6


@dataclass
class FileMetrics:
    rel_path: str
    file_name: str
    model_name: str
    n_options: Optional[int]
    total_questions: int
    total_correct: int
    accuracy: float
    bss: Optional[float]
    skill_score: Optional[float]
    q_acc: Dict[str, float]
    q_total: Dict[str, int]
    q_correct: Dict[str, int]
    cluster_q1_3_acc: float
    cluster_q4_6_acc: float
    cluster_q1_3_correct: int
    cluster_q1_3_total: int
    cluster_q4_6_correct: int
    cluster_q4_6_total: int
    cluster_gap_q4_6_minus_q1_3: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark result suite with visualizations")
    benchmark_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=benchmark_dir / "result",
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=benchmark_dir / "analysis_suite",
        help="Directory to save analysis CSVs and plots",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Max question index to analyze per video",
    )
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def parse_num_options(file_name: str) -> Optional[int]:
    match = re.search(r"generated_options_(\d+)opts", file_name)
    if not match:
        return None
    return int(match.group(1))


def compute_chance_adjusted_scores(acc: float, n_options: Optional[int]) -> tuple[Optional[float], Optional[float]]:
    if not n_options or n_options <= 1:
        return None, None

    chance = 1.0 / n_options
    bss = acc * (1.0 - chance)
    # Skill score in [0,1] where 0 means chance-level, 1 means perfect.
    skill = safe_div(acc - chance, 1.0 - chance)
    return bss, skill


def parse_one_result_file(json_path: Path, result_root: Path, max_questions: int) -> Optional[FileMetrics]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] Skip invalid JSON {json_path}: {exc}")
        return None

    model_name = data.get("model_name", json_path.stem.replace("results_", ""))
    records = data.get("results", [])
    if not isinstance(records, list) or not records:
        print(f"[WARN] Empty result list: {json_path}")
        return None

    q_total = {f"Q{i}": 0 for i in range(1, max_questions + 1)}
    q_correct = {f"Q{i}": 0 for i in range(1, max_questions + 1)}

    for row in records:
        choices = row.get("choices", [])
        correct = row.get("correct", [])
        if not isinstance(choices, list) or not isinstance(correct, list):
            continue

        valid_len = min(len(choices), len(correct), max_questions)
        for idx in range(valid_len):
            q_key = f"Q{idx + 1}"
            q_total[q_key] += 1
            if choices[idx] == correct[idx]:
                q_correct[q_key] += 1

    total_questions = sum(q_total.values())
    total_correct = sum(q_correct.values())
    accuracy = safe_div(total_correct, total_questions)

    n_options = parse_num_options(json_path.name)
    bss, skill = compute_chance_adjusted_scores(accuracy, n_options)

    q_acc = {q: safe_div(q_correct[q], q_total[q]) for q in q_total}

    q1_3_correct = q_correct["Q1"] + q_correct["Q2"] + q_correct["Q3"]
    q1_3_total = q_total["Q1"] + q_total["Q2"] + q_total["Q3"]
    q4_6_correct = q_correct["Q4"] + q_correct["Q5"] + q_correct["Q6"]
    q4_6_total = q_total["Q4"] + q_total["Q5"] + q_total["Q6"]

    cluster_q1_3_acc = safe_div(q1_3_correct, q1_3_total)
    cluster_q4_6_acc = safe_div(q4_6_correct, q4_6_total)
    cluster_gap = cluster_q4_6_acc - cluster_q1_3_acc

    return FileMetrics(
        rel_path=str(json_path.relative_to(result_root)).replace("\\", "/"),
        file_name=json_path.name,
        model_name=model_name,
        n_options=n_options,
        total_questions=total_questions,
        total_correct=total_correct,
        accuracy=accuracy,
        bss=bss,
        skill_score=skill,
        q_acc=q_acc,
        q_total=q_total,
        q_correct=q_correct,
        cluster_q1_3_acc=cluster_q1_3_acc,
        cluster_q4_6_acc=cluster_q4_6_acc,
        cluster_q1_3_correct=q1_3_correct,
        cluster_q1_3_total=q1_3_total,
        cluster_q4_6_correct=q4_6_correct,
        cluster_q4_6_total=q4_6_total,
        cluster_gap_q4_6_minus_q1_3=cluster_gap,
    )


def collect_metrics(result_dir: Path, max_questions: int) -> pd.DataFrame:
    rows: List[Dict] = []
    files = sorted(result_dir.rglob("results_*.json"))

    for json_path in files:
        metrics = parse_one_result_file(json_path, result_dir, max_questions)
        if not metrics:
            continue

        row = {
            "rel_path": metrics.rel_path,
            "file_name": metrics.file_name,
            "model_name": metrics.model_name,
            "n_options": metrics.n_options,
            "accuracy": metrics.accuracy,
            "bss": metrics.bss,
            "skill_score": metrics.skill_score,
            "total_correct": metrics.total_correct,
            "total_questions": metrics.total_questions,
            "cluster_q1_3_acc": metrics.cluster_q1_3_acc,
            "cluster_q4_6_acc": metrics.cluster_q4_6_acc,
            "cluster_q1_3_correct": metrics.cluster_q1_3_correct,
            "cluster_q1_3_total": metrics.cluster_q1_3_total,
            "cluster_q4_6_correct": metrics.cluster_q4_6_correct,
            "cluster_q4_6_total": metrics.cluster_q4_6_total,
            "cluster_gap_q4_6_minus_q1_3": metrics.cluster_gap_q4_6_minus_q1_3,
        }
        for q_idx in range(1, max_questions + 1):
            q_key = f"Q{q_idx}"
            row[f"{q_key}_acc"] = metrics.q_acc[q_key]
            row[f"{q_key}_correct"] = metrics.q_correct[q_key]
            row[f"{q_key}_total"] = metrics.q_total[q_key]

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["n_options", "accuracy"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return df


def canonicalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate records by (model_name, n_options).

    Priority:
    1) Prefer records outside phase2/
    2) Shallower relative path
    3) Lexicographically smaller file name
    """
    if df.empty:
        return df

    work = df.copy()
    work["_is_phase2"] = work["rel_path"].str.contains(r"(?:^|/)phase2/", regex=True)
    work["_path_depth"] = work["rel_path"].str.count("/")

    work = work.sort_values(
        ["model_name", "n_options", "_is_phase2", "_path_depth", "file_name"],
        ascending=[True, True, True, True, True],
    )
    work = work.drop_duplicates(subset=["model_name", "n_options"], keep="first")
    work = work.drop(columns=["_is_phase2", "_path_depth"]).reset_index(drop=True)
    return work


def save_csv_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    paths = {}

    all_csv = output_dir / "all_results_metrics.csv"
    df.to_csv(all_csv, index=False)
    paths["all_results_metrics"] = all_csv

    canonical_df = canonicalize_records(df)
    canonical_csv = output_dir / "canonical_results_metrics.csv"
    canonical_df.to_csv(canonical_csv, index=False)
    paths["canonical_results_metrics"] = canonical_csv

    df_5 = canonical_df[canonical_df["n_options"] == 5].copy()
    if not df_5.empty:
        leaderboard_5 = df_5.sort_values("accuracy", ascending=False)
        leaderboard_5_csv = output_dir / "leaderboard_5opts.csv"
        leaderboard_5.to_csv(leaderboard_5_csv, index=False)
        paths["leaderboard_5opts"] = leaderboard_5_csv

        cluster_5 = leaderboard_5[[
            "model_name",
            "accuracy",
            "cluster_q1_3_acc",
            "cluster_q4_6_acc",
            "cluster_gap_q4_6_minus_q1_3",
        ]].copy()
        cluster_5_csv = output_dir / "cluster_stats_5opts.csv"
        cluster_5.to_csv(cluster_5_csv, index=False)
        paths["cluster_stats_5opts"] = cluster_5_csv

    option_curve = (
        canonical_df.dropna(subset=["n_options"])
        .groupby(["model_name", "n_options"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            bss=("bss", "mean"),
            skill_score=("skill_score", "mean"),
            cluster_q1_3_acc=("cluster_q1_3_acc", "mean"),
            cluster_q4_6_acc=("cluster_q4_6_acc", "mean"),
        )
        .sort_values(["model_name", "n_options"])
    )
    option_curve_csv = output_dir / "option_sensitivity_summary.csv"
    option_curve.to_csv(option_curve_csv, index=False)
    paths["option_sensitivity_summary"] = option_curve_csv

    return paths


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300


def plot_leaderboard_5opts(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    df = canonicalize_records(df)
    df_5 = df[df["n_options"] == 5].sort_values("accuracy", ascending=True)
    if df_5.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    y = list(range(len(df_5)))
    bar_h = 0.36
    ax.barh([v + bar_h / 2 for v in y], df_5["accuracy"], height=bar_h, alpha=0.9, label="Accuracy")
    ax.barh([v - bar_h / 2 for v in y], df_5["bss"], height=bar_h, alpha=0.8, label="BSS")

    for i, row in enumerate(df_5.itertuples(index=False)):
        ax.text(row.accuracy + 0.005, i + bar_h / 2, f"{row.accuracy:.3f}", va="center", fontsize=10)
        if row.bss is not None and not math.isnan(row.bss):
            ax.text(row.bss + 0.005, i - bar_h / 2, f"{row.bss:.3f}", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(df_5["model_name"])
    ax.set_xlabel("Score")
    ax.set_title("5-Option Leaderboard: Accuracy vs BSS")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = output_dir / "plot_01_leaderboard_5opts_accuracy_bss.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_option_sensitivity(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    df = canonicalize_records(df)
    option_df = (
        df.dropna(subset=["n_options"])
        .groupby(["model_name", "n_options"], as_index=False)["accuracy"]
        .mean()
    )

    # Keep models that have at least 2 option points.
    point_count = option_df.groupby("model_name")["n_options"].nunique()
    kept = point_count[point_count >= 2].index.tolist()
    option_df = option_df[option_df["model_name"].isin(kept)]

    if option_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=option_df,
        x="n_options",
        y="accuracy",
        hue="model_name",
        marker="o",
        linewidth=2,
        ax=ax,
    )

    ax.set_title("Option Sensitivity: Accuracy vs Number of Options")
    ax.set_xlabel("Number of Options (N)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(option_df["n_options"].unique().tolist()))
    ax.legend(title="Model", loc="best")
    fig.tight_layout()

    out = output_dir / "plot_02_option_sensitivity_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cluster_bars_5opts(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    df = canonicalize_records(df)
    df_5 = df[df["n_options"] == 5].copy()
    if df_5.empty:
        return None

    df_5 = df_5.sort_values("accuracy", ascending=False)
    melt_df = df_5[["model_name", "cluster_q1_3_acc", "cluster_q4_6_acc"]].melt(
        id_vars=["model_name"],
        value_vars=["cluster_q1_3_acc", "cluster_q4_6_acc"],
        var_name="cluster",
        value_name="accuracy",
    )
    label_map = {
        "cluster_q1_3_acc": "Q1-3 Cluster",
        "cluster_q4_6_acc": "Q4-6 Cluster",
    }
    melt_df["cluster"] = melt_df["cluster"].map(label_map)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=melt_df, x="model_name", y="accuracy", hue="cluster", ax=ax)
    ax.set_title("5-Option Cluster Statistics: Q1-3 vs Q4-6")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Question Cluster")
    fig.tight_layout()

    out = output_dir / "plot_03_cluster_q1_3_vs_q4_6_5opts.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_q1_q6_heatmap_5opts(df: pd.DataFrame, output_dir: Path, max_questions: int) -> Optional[Path]:
    df = canonicalize_records(df)
    df_5 = df[df["n_options"] == 5].copy()
    if df_5.empty:
        return None

    q_cols = [f"Q{i}_acc" for i in range(1, max_questions + 1)]
    heat_df = df_5[["model_name", *q_cols]].sort_values("model_name").set_index("model_name")

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("5-Option Per-Question Accuracy Heatmap (Q1-Q6)")
    ax.set_xlabel("Question")
    ax.set_ylabel("Model")
    fig.tight_layout()

    out = output_dir / "plot_04_q1_q6_heatmap_5opts.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_markdown_report(df: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "ANALYSIS_SUMMARY.md"

    lines: List[str] = []
    lines.append("# Benchmark Analysis Summary")
    lines.append("")
    lines.append("## 1) Data Coverage")
    lines.append(f"- Result files analyzed: {len(df)}")
    if not df.empty:
        lines.append(f"- Models covered: {df['model_name'].nunique()}")
        options = sorted([int(x) for x in df['n_options'].dropna().unique().tolist()])
        lines.append(f"- Option counts covered: {options}")
    lines.append("")

    if not df.empty:
        canonical_df = canonicalize_records(df)
        df_5 = canonical_df[canonical_df["n_options"] == 5].sort_values("accuracy", ascending=False)
        lines.append("## 2) 5-Option Leaderboard")
        if df_5.empty:
            lines.append("- No 5-option results found.")
        else:
            lines.append("| Rank | Model | Accuracy | BSS | SkillScore | Q1-3 | Q4-6 | Gap(Q4-6-Q1-3) |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
            for idx, row in enumerate(df_5.itertuples(index=False), start=1):
                bss_txt = "-" if row.bss is None or math.isnan(row.bss) else f"{row.bss:.4f}"
                skill_txt = "-" if row.skill_score is None or math.isnan(row.skill_score) else f"{row.skill_score:.4f}"
                lines.append(
                    f"| {idx} | {row.model_name} | {row.accuracy:.4f} | {bss_txt} | {skill_txt} | "
                    f"{row.cluster_q1_3_acc:.4f} | {row.cluster_q4_6_acc:.4f} | {row.cluster_gap_q4_6_minus_q1_3:.4f} |"
                )
        lines.append("")

        lines.append("## 3) Cluster Interpretation")
        lines.append("- `Q1-3` cluster: early-stage question group")
        lines.append("- `Q4-6` cluster: later-stage question group")
        lines.append("- Positive gap means model performs better on Q4-6 than Q1-3")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    print("=" * 70)
    print("Benchmark Result Suite Analysis")
    print("=" * 70)
    print(f"Result dir : {result_dir}")
    print(f"Output dir : {output_dir}")

    df = collect_metrics(result_dir=result_dir, max_questions=args.max_questions)
    if df.empty:
        print("[ERROR] No valid result JSON files found.")
        return

    csv_paths = save_csv_tables(df, output_dir)

    setup_plot_style()
    plot_paths = [
        plot_leaderboard_5opts(df, output_dir),
        plot_option_sensitivity(df, output_dir),
        plot_cluster_bars_5opts(df, output_dir),
        plot_q1_q6_heatmap_5opts(df, output_dir, args.max_questions),
    ]

    report_path = save_markdown_report(df, output_dir)

    print("\n[Done] CSV files:")
    for name, path in csv_paths.items():
        print(f"- {name}: {path}")

    print("\n[Done] Plot files:")
    for path in plot_paths:
        if path:
            print(f"- {path}")

    print(f"\n[Done] Summary report: {report_path}")


if __name__ == "__main__":
    main()
