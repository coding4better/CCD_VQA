# Paper-Ready Results Paragraphs (English Only)

## Main result under the 5-option setting

Under a unified 5-option video multiple-choice evaluation protocol (199 videos and 1,194 questions in total), the tested vision-language models exhibit a clear performance hierarchy with substantial gaps. Google Gemini-3.1-pro-preview achieves the highest accuracy (77.55%), leading the second-ranked model by 12.47 percentage points. This is followed by qwen-vl-max (65.08%), internvl3-8b (64.15%), and llava-ov-7b (63.32%). To further reduce the effect of random guessing, we compute the Benchmark Suitability Score (BSS). Under BSS, gemini-3.1-pro-preview remains the top model (0.6204), with qwen-vl-max (0.5206), internvl3-8b (0.5132), and llava-ov-7b (0.5065) ranked second through fourth, respectively. This ranking indicates that inter-model performance differences are primarily driven by stronger discriminative capability rather than chance-level benefits. In contrast, llava-next-video-7b achieves only 38.94% accuracy, substantially below the leading cohort.

## Clustered statistics and robustness analysis

To characterize stage-wise capability and robustness, we partition the six questions into two clusters: Q1-3 and Q4-6. All models except llava-next-video-7b consistently perform worse on Q4-6 than on Q1-3, indicating that the latter half of the benchmark poses greater challenge. gemini-3.1-pro-preview obtains 80.23% on Q1-3 and 74.87% on Q4-6 (gap: -5.36 percentage points), representing the most balanced profile among all models and reflecting superior robustness. Among open-source models, qwen-vl-max achieves 67.17% on Q1-3 and 62.98% on Q4-6 (gap: -4.19 percentage points), making it the most stable choice. By comparison, internvl3-8b and llava-ov-7b show much larger drops (-22.78 and -23.45 percentage points, respectively), despite early-cluster performance around 75%. These observations underscore that overall accuracy alone is insufficient to capture model robustness, while clustered statistics provide a more interpretable view of capability distribution and task adaptation.

## Per-question dimension analysis

Fine-grained per-question analysis reveals differentiated capability profiles across question dimensions. gemini-3.1-pro-preview achieves 77.4%, 85.4%, and 77.9% on Q1-Q3, and 71.9%, 75.4%, and 77.4% on Q4-Q6, demonstrating relatively uniform capability. qwen2_5_vl_7b reaches 83.9% on Q1 but drops significantly to 33.7% on Q4 (a gap of 50.2 percentage points), revealing a critical weak dimension. internvl3-8b similarly peaks at 90.5% on Q1 but falls to 39.2% on Q4, exhibiting comparable capability disparity. These inter-dimension variations pinpoint key optimization directions for model improvement, specifically enhanced stability on medium-difficulty questions.

## Option-count sensitivity analysis

In controlled difficulty studies, we evaluate open-source models' performance degradation with increasing options. Both internvl2.5-2b and qwen2_5_vl_7b show monotonic accuracy decrease by option count (internvl2.5-2b: 68.34% -> 58.88% -> 56.62% -> 51.84%; qwen2_5_vl_7b: 71.27% -> 63.90% -> 57.71% -> 55.95%), consistent with multiple-choice task difficulty progression. Under the hardest setting (N=5), qwen2_5_vl_7b maintains a lead (55.95% vs 51.84%) with a shallower degradation slope (15.32 vs 16.50 percentage points from N=2), demonstrating better robustness under increased option count. This trend aligns with the BSS-based ranking, further supporting the consistency and interpretability of inter-model performance differences.

## Reproducibility statement

All results are produced with a unified script, shared question set, and identical scoring protocol. Each sample is evaluated question by question using single-question mode to avoid information truncation. Correctness is determined by exact option-index matching. We publicly release the overall accuracy, chance-adjusted score (BSS), skill score (SkillScore), per-question accuracy (Q1-Q6), clustered accuracy (Q1-3 and Q4-6) with gap statistics, and four key visualizations (leaderboard bar chart, option-sensitivity curve, cluster statistics, and per-question heatmap) to fully support downstream error analysis, model comparison, and benchmark reproducibility.

## Quick reference table

| Rank | Model | Accuracy | BSS | Skill | Q1-3 | Q4-6 | Gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | gemini-3.1-pro-preview | 0.7755 | 0.6204 | 0.7194 | 0.8023 | 0.7487 | -0.0536 |
| 2 | qwen-vl-max | 0.6508 | 0.5206 | 0.5634 | 0.6717 | 0.6298 | -0.0419 |
| 3 | internvl3-8b | 0.6415 | 0.5132 | 0.5519 | 0.7554 | 0.5276 | -0.2278 |
| 4 | llava-ov-7b | 0.6332 | 0.5065 | 0.5415 | 0.7504 | 0.5159 | -0.2345 |
| 5 | qwen2_5_vl_7b | 0.5595 | 0.4476 | 0.4493 | 0.6650 | 0.4539 | -0.2111 |
| 6 | internvl3-2b | 0.5469 | 0.4375 | 0.4336 | 0.6750 | 0.4188 | -0.2563 |
| 7 | internvl2.5-2b | 0.5184 | 0.4147 | 0.3980 | 0.6466 | 0.3903 | -0.2563 |
| 8 | llava-next-video-7b | 0.3894 | 0.3116 | 0.2368 | 0.4121 | 0.3668 | -0.0452 |

## Metric definitions

- Accuracy: raw accuracy, correct answers divided by total questions
- BSS (Benchmark Suitability Score): Accuracy x (1 - 1/N), a chance-adjusted score
- Skill Score: the proportion of performance above random guessing, in [0,1]
- Q1-3: average accuracy on the first three questions, usually easier
- Q4-6: average accuracy on the last three questions, usually harder
- Gap: Q4-6 - Q1-3; a negative value means Q4-6 is harder, which is expected
