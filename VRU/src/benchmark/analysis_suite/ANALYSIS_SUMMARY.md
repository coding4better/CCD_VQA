# Benchmark Analysis Summary

## 1) Data Coverage
- Result files analyzed: 16
- Models covered: 8
- Option counts covered: [2, 3, 4, 5]

## 2) 5-Option Leaderboard
| Rank | Model | Accuracy | BSS | SkillScore | Q1-3 | Q4-6 | Gap(Q4-6-Q1-3) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | gemini-3.1-pro-preview | 0.7755 | 0.6204 | 0.7194 | 0.8023 | 0.7487 | -0.0536 |
| 2 | qwen-vl-max | 0.6508 | 0.5206 | 0.5634 | 0.6717 | 0.6298 | -0.0419 |
| 3 | internvl3-8b | 0.6415 | 0.5132 | 0.5519 | 0.7554 | 0.5276 | -0.2278 |
| 4 | llava-ov-7b | 0.6332 | 0.5065 | 0.5415 | 0.7504 | 0.5159 | -0.2345 |
| 5 | qwen2_5_vl_7b | 0.5595 | 0.4476 | 0.4493 | 0.6650 | 0.4539 | -0.2111 |
| 6 | internvl3-2b | 0.5469 | 0.4375 | 0.4336 | 0.6750 | 0.4188 | -0.2563 |
| 7 | internvl2.5-2b | 0.5184 | 0.4147 | 0.3980 | 0.6466 | 0.3903 | -0.2563 |
| 8 | llava-next-video-7b | 0.3894 | 0.3116 | 0.2368 | 0.4121 | 0.3668 | -0.0452 |

## 3) Cluster Interpretation
- `Q1-3` cluster: early-stage question group
- `Q4-6` cluster: later-stage question group
- Positive gap means model performs better on Q4-6 than Q1-3
