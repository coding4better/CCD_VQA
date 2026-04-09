# Benchmark - Video QA Evaluation

Evaluate multimodal model video understanding on the crash dataset.

## Quick Start

### 1. Download and run the default models

If you want to run InternVL3-2B and InternVL3-8B directly, use the batch script to download and infer in one step:

```bash
bash batch_download_and_run.sh
```

This downloads `OpenGVLab/InternVL3-2B` and `OpenGVLab/InternVL3-8B` first, then runs them one by one.

### Runtime configuration

The table below summarizes the effective settings used by `batch_download_and_run.sh` and `run_benchmark_v2.py`.

| Item | Current value | Meaning | Editable? |
|---|---:|---|---|
| Default models | `internvl3-2b,internvl3-8b` | Only these two InternVL3 variants are run | Yes |
| Model parallelism | `0` | Runs sequentially to avoid GPU and system contention | Not recommended |
| Max frames per video | `32` | Extract up to 32 frames per video before inference | Yes |
| Frame sampling rate | `5 FPS` | Sampling rate used by `load_video_frames()` | Yes |
| InternVL3 input frames | `10` | `InternVLRunner.predict(..., num_frames=10)` resamples to at most 10 frames | Yes |
| Single-question mode | Enabled | Each video is answered question by question | Not recommended |
| Answers returned each step | `1` | `expected_count=1`, so only one option index is parsed each time | Not recommended |
| Option count | Dynamic | Pulled from the CSV, typically 3/4/5 options | Not recommended |
| Model download root | `/root/autodl-tmp/hf_models` | Downloads go to the data disk instead of system cache | Yes |
| Result directory | `result/` | JSON results are saved per model | Not recommended |
| Existing result handling | Auto-skip | Existing result files are reused instead of rerun | Not recommended |

To temporarily switch the model list, override it with an environment variable:

```bash
MODELS_CSV="internvl3-2b" bash batch_download_and_run.sh
```

If you want to hard-code a model list, edit `get_model_list()` in `models/model_zoo.py`:

```python
def get_model_list():
    return [
        'internvl3-2b',
        'internvl3-8b',
    ]
```

### 2. Run evaluation

```bash
cd /home/24068286g/UString
python VRU/src/benchmark/run_benchmark.py
```

If you only want to evaluate the current InternVL3 setup, run the batch script inside `VRU/src/benchmark/` directly:

```bash
bash batch_download_and_run.sh
```

## Supported models

### Local models
- **InternVL series**: `internvl2-2b`, `internvl2-4b`, `internvl2-8b`, `internvl2.5-1b/2b/4b/8b`, `internvl3-1b/2b/4b/8b`
- **LLaVA**: `llava`, `llava-next-7b`
- **Qwen**: `qwen2-vl-7b`

### API models (API key required)
- **Gemini**: `gemini-api` (`GEMINI_API_KEY`)
- **Qwen API**: `qwen-api` (`DASHSCOPE_API_KEY`)

## Directory Layout

```
benchmark/
├── run_benchmark.py          # Main runner
├── models/
│   ├── model_zoo.py         # Model factory
│   ├── llava_runner.py       # LLaVA inference
│   ├── internvl_runner.py    # InternVL inference
│   ├── qwen_runner.py       # Qwen inference
│   ├── gemini_runner.py      # Gemini API inference
│   ├── base_runner.py        # Base runner
│   └── __init__.py
└── results/                  # Evaluation results
    ├── results_llava.json
    ├── results_internvl2.5-4b.json
    └── ...
```

## Result Format

Results are stored as JSON and include overall accuracy plus video-level and question-level metrics:

```json
{
  "model_name": "internvl2.5-4b",
  "overall_accuracy": 0.75,
  "total_correct": 180,
  "total_questions": 240,
  "num_videos": 10,
  "results": [...]
}
```

## Analysis and Visualization

The unified analysis script `analyze_result_suite.py` can produce:

- Aggregation across all results, including `result/` and nested folders such as `phase2/`
- Leaderboard metrics: Accuracy / BSS / SkillScore
- Per-question statistics: Q1-Q6
- Cluster statistics: Q1-3 versus Q4-6, including gaps
- CSV exports and PNG visualizations

Run it with:

```bash
cd VRU/src/benchmark
python analyze_result_suite.py --result-dir result --output-dir analysis_suite
```

Default outputs:

- `analysis_suite/all_results_metrics.csv`
- `analysis_suite/leaderboard_5opts.csv`
- `analysis_suite/cluster_stats_5opts.csv`
- `analysis_suite/option_sensitivity_summary.csv`
- `analysis_suite/plot_01_leaderboard_5opts_accuracy_bss.png`
- `analysis_suite/plot_02_option_sensitivity_accuracy.png`
- `analysis_suite/plot_03_cluster_q1_3_vs_q4_6_5opts.png`
- `analysis_suite/plot_04_q1_q6_heatmap_5opts.png`
- `analysis_suite/ANALYSIS_SUMMARY.md`

## Runner Interface

Custom runners should implement:

```python
class MyRunner:
    def __init__(self, model_name: str):
        """Initialize the model."""
        pass
    
    def predict(self, video_number: str, prompt: str, video_frames: np.ndarray) -> list:
        """
        Run multiple-choice inference.
        
        Args:
            video_number: Video ID, for example "000003"
            prompt: Full prompt including the question and answer options
            video_frames: Video frame array with shape=(num_frames, H, W, 3)
        
        Returns:
            list: A length-N list, each element is an option index from 1 to K
        """
        pass
    
    def release(self):
        """Release model resources."""
        pass
```

## Dataset

Evaluation uses the questions and options from `VRU/src/option_generate/data/QA_pair_v2_4options.csv`.
