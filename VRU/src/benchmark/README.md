# Benchmark - 视频QA评估

在crash数据集上评估多模态大模型的视频理解能力。

## 快速开始

### 1. 配置要运行的模型

编辑 `models/model_zoo.py` 中的 `get_model_list()`，选择要评估的模型：

```python
def get_model_list():
    return [
        'internvl2.5-4b',      # 推荐
        'llava',
    ]
```

### 2. 运行评估

```bash
cd /home/24068286g/UString
python VRU/src/benchmark/run_benchmark.py
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
