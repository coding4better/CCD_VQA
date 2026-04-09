# CCD_VQA

This repository contains the code, analysis tools, and benchmark outputs for a crash and traffic-scene visual question answering workflow.

The active project lives under `VRU/`. The root-level README is kept as a compact map of the current folder structure so the repository matches the actual working content.

## Repository Layout

### `VRU/src/`
Main source tree for the project.

- `benchmark/`: model evaluation, result parsing, and visualization utilities
- `ablation_study/`: paper notes, configuration references, and analysis reports
- `description_generation/`: video description generation utilities and API guides
- `description_check/`: consistency checking helpers
- `option_generate/`: option generation pipeline and data files
- `threshold_analysis/`: threshold-related analysis scripts and outputs
- `video_filtering/`: video filtering and preprocessing utilities
- `data_exploration/`: exploratory notebooks and scripts
- `dataset_analysis/`: dataset inspection and visualization artifacts
- `api_test_framework/`: modular API testing pipeline for model inference

### `VRU/vid_list/`
Prepared video lists, generated QA datasets, and standardized CSV/JSON files used by the benchmark pipeline.

### `requirements.txt`
Python dependencies for the current environment.

## Main Workflows

### Benchmarking

The benchmark suite evaluates video understanding models on multiple-choice crash-scene questions. The main entry points are:

- `VRU/src/benchmark/run_benchmark.py`
- `VRU/src/benchmark/batch_download_and_run.sh`
- `VRU/src/benchmark/analyze_result_suite.py`

The latest 5-option benchmark outputs and summary tables are stored under `VRU/src/benchmark/result/` and `VRU/src/benchmark/analysis_suite/`.

### Question and Option Generation

Question, answer, and option generation utilities are grouped under:

- `VRU/src/description_generation/`
- `VRU/src/option_generate/`
- `VRU/src/description_check/`

### Filtering and Analysis

Video filtering and threshold analysis utilities are located in:

- `VRU/src/video_filtering/`
- `VRU/src/threshold_analysis/`
- `VRU/src/data_exploration/`

## Recommended Starting Points

If you are new to the repository, start with these files:

1. `VRU/src/benchmark/README.md` for evaluation usage and result formats
2. `VRU/src/description_generation/GEMINI_API_GUIDE.md` for Gemini-based generation setup
3. `VRU/src/ablation_study/VQA_Dataset_Configuration_Quick_Reference.md` for the option-configuration summary
4. `VRU/src/benchmark/analysis_suite/ANALYSIS_SUMMARY.md` for the latest benchmark summary

## Notes

- The root-level legacy `src/`, `demo.py`, and `main.py` files were removed in the `english-version` branch to keep the repository focused on the current VRU workflow.
- Most current documentation and report files under `VRU/src/` are written in English for publication and sharing.
