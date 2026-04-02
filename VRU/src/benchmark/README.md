# Benchmark - 视频QA评估

在crash数据集上评估多模态大模型的视频理解能力。

## 快速开始

### 1. 下载并配置要运行的模型

如果你要直接跑 InternVL3-2B 和 InternVL3-8B，推荐先用批处理脚本完成下载和推理：

```bash
bash batch_download_and_run.sh
```

默认会先下载 `OpenGVLab/InternVL3-2B` 和 `OpenGVLab/InternVL3-8B`，然后逐个执行推理。

### 运行参数总表

下表是当前 `batch_download_and_run.sh` + `run_benchmark_v2.py` 的实际生效配置总览：

| 参数项 | 当前值 | 说明 | 可改/不建议改 |
|---|---:|---|---|
| 默认模型 | `internvl3-2b,internvl3-8b` | 只跑这两个 InternVL3 版本 | 可改 |
| 模型并行 | `0` | 串行执行，避免同时占用显存和系统资源 | 不建议改 |
| 视频抽帧上限 | `32` | 每个视频先统一抽取最多 32 帧 | 可改 |
| 抽帧速率 | `5 FPS` | `load_video_frames()` 的采样速率 | 可改 |
| InternVL3 实际输入帧数 | `10` | `InternVLRunner.predict(..., num_frames=10)` 默认再采样到最多 10 帧 | 可改 |
| 单题模式 | 开启 | 每个视频按题目逐题推理，而不是一次性回答整组题目 | 不建议改 |
| 每次返回答案数 | `1` | 当前 `expected_count=1`，每次只解析一个选项编号 | 不建议改 |
| 选项数 | 动态 | 按 CSV 里的题目选项数传入，通常是 3/4/5 | 不建议改 |
| 模型下载根目录 | `/root/autodl-tmp/hf_models` | 默认下载到数据盘，避免写入系统盘缓存 | 可改 |
| 结果目录 | `result/` | 每个模型单独保存 JSON 结果 | 不建议改 |
| 已完成结果处理 | 自动跳过 | 若结果文件已存在，则直接复用，不重复跑 | 不建议改 |

如果你要临时切换模型列表，可以通过环境变量覆盖：

```bash
MODELS_CSV="internvl3-2b" bash batch_download_and_run.sh
```

如果只想手动指定模型，也可以编辑 `models/model_zoo.py` 中的 `get_model_list()`。

手动指定时可以这样写：

```python
def get_model_list():
    return [
        'internvl3-2b',
        'internvl3-8b',
    ]
```

### 2. 运行评估

```bash
cd /home/24068286g/UString
python VRU/src/benchmark/run_benchmark.py
```

如果你只评测当前这套 InternVL3 配置，直接在 `VRU/src/benchmark/` 下运行：

```bash
bash batch_download_and_run.sh
```

## 支持的模型

### 本地模型
- **InternVL系列**: `internvl2-2b`, `internvl2-4b`, `internvl2-8b`, `internvl2.5-1b/2b/4b/8b`, `internvl3-1b/2b/4b/8b`
- **LLaVA**: `llava`, `llava-next-7b`
- **Qwen**: `qwen2-vl-7b`

### API模型（需配置API Key）
- **Gemini**: `gemini-api` (需环境变量: `GEMINI_API_KEY`)
- **Qwen API**: `qwen-api` (需环境变量: `DASHSCOPE_API_KEY`)

## 目录结构

```
benchmark/
├── run_benchmark.py          # 主运行脚本
├── models/
│   ├── model_zoo.py         # 模型工厂
│   ├── llava_runner.py       # LLaVA 推理
│   ├── internvl_runner.py    # InternVL 推理
│   ├── qwen_runner.py        # Qwen 推理
│   ├── gemini_runner.py      # Gemini API 推理
│   ├── base_runner.py        # 基础 Runner
│   └── __init__.py
└── results/                  # 评估结果
    ├── results_llava.json
    ├── results_internvl2.5-4b.json
    └── ...
```

## 结果格式

结果保存为JSON格式，包含整体准确率、视频级别和问题级别的评估：

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

## Runner接口

自定义模型需实现的接口：

```python
class MyRunner:
    def __init__(self, model_name: str):
        """初始化模型"""
        pass
    
    def predict(self, video_number: str, prompt: str, video_frames: np.ndarray) -> list:
        """
        推理选择题
        
        Args:
            video_number: 视频编号（如"000003"）
            prompt: 包含题干和选项的完整prompt
            video_frames: 视频帧数组，shape=(num_frames, H, W, 3)
        
        Returns:
            list: 长度为N的列表，每个元素为1-K（选项序号）
        """
        pass
    
    def release(self):
        """释放模型资源"""
        pass
```

## 数据集

使用 `VRU/src/option_generate/data/QA_pair_v2_4options.csv` 中的问题和选项进行评估。
