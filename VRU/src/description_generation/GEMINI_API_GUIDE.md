# Gemini API 使用完整指南

## 🔑 第一步：获取 Gemini API 密钥

### 1.1 获取密钥
访问: https://aistudio.google.com/apikey

步骤：
1. 用 Google 账号登录
2. 点击 "Create API Key"
3. 选择 "Create new secret key"
4. 复制生成的密钥

### 1.2 检查密钥有效性
```bash
# 测试密钥是否有效
curl https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY
```

---

## 🔐 第二步：在代码中设置密钥

### 方法 A：修改 gemini_runner.py（推荐）

编辑文件：`/home/24068286g/UString/VRU/src/benchmark/models/gemini_runner.py`

找到第 13 行：
```python
self.api_key = "your_gemini_api_key_here"
```

替换为你的实际密钥：
```python
self.api_key = "AIza...your_actual_key..."
```

### 方法 B：使用环境变量

```bash
export GEMINI_API_KEY="your_actual_api_key"
```

然后 GeminiRunner 会自动读取。

### 方法 C：在脚本中动态设置

```python
from benchmark.models.gemini_runner import GeminiRunner

runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_actual_api_key"
runner._init_api()  # 重新初始化
```

---

## 🚀 第三步：运行推理工作流

### 3.1 使用完整工作流脚本

```bash
cd /home/24068286g/CCD_VQA/VRU/src/description_generation

# 编辑脚本，设置 API 密钥
# 然后运行
python gemini_workflow.py
```

### 3.2 快速测试（只测试 1 条数据）

```python
from description_generation import load_qa_data
from api_test_framework.video_processor import load_video_frames
from benchmark.models.gemini_runner import GeminiRunner

# 1. 加载数据
data = load_qa_data("/path/to/csv")
item = data[0]

# 2. 设置 Gemini
runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

# 3. 构建 Prompt
prompt = f"""Based on:\n{item['facts_text']}\n\nGenerate a description."""

# 4. 推理
result = runner.predict(item['video_id'], prompt, None)  # None = 纯文本推理
print(result)
```

### 3.3 包含视频帧的完整推理

```python
from description_generation import load_qa_data
from api_test_framework.video_processor import load_video_frames
from benchmark.models.gemini_runner import GeminiRunner

# 1. 加载数据和视频
data = load_qa_data("/path/to/csv")
item = data[0]
video_frames = load_video_frames("/path/to/video.mp4", max_frames=16, target_fps=6)

# 2. 初始化 Gemini
runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

# 3. Prompt
prompt = f"""Based on:\n{item['facts_text']}\n\nGenerate a description."""

# 4. 推理（带视频帧）
result = runner.predict(item['video_id'], prompt, video_frames)
print(result)
```

---

## ⚙️ 配置参数

### Gemini 模型选择

```python
# 可用的模型（按优先级）
models = [
    "gemini-2.5-pro",      # 最新、最强
    "gemini-2.0-pro",      # 稳定、强大
    "gemini-pro",          # 通用模型
]

runner = GeminiRunner("gemini-2.5-pro")
```

### 推理参数调整

```python
# 在 predict() 方法中调整
response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,          # 0-1，越低越确定，越高越创意
        max_output_tokens=512,    # 最多生成多少 token
        top_p=0.9,               # 核心采样参数
        top_k=40,                # Top-K 采样
    )
)
```

### 视频帧参数调整

```python
# 帧采样参数
video_frames = load_video_frames(
    video_path,
    max_frames=32,    # 最多提取多少帧
    target_fps=10     # 目标采样速率
)
```

---

## 🐛 常见问题和解决方案

### 问题 1: 地理位置限制错误

**错误信息**:
```
Location not supported for the API use
```

**解决方案**:
```bash
# 配置代理
export HTTP_PROXY="http://proxy:port"
export HTTPS_PROXY="http://proxy:port"

# 或使用 VPN
# 或在允许的地区运行
```

### 问题 2: API 密钥无效

**错误信息**:
```
INVALID_ARGUMENT: API key not valid
Invalid API Key
```

**解决方案**:
1. 检查密钥是否正确复制
2. 检查密钥是否已启用
3. 重新生成新的密钥

### 问题 3: 配额已用尽

**错误信息**:
```
RESOURCE_EXHAUSTED: Quota exceeded
RATE_LIMIT: Please retry after
```

**解决方案**:
1. 等待配额重置（通常是每天）
2. 升级到付费计划
3. 降低请求频率

### 问题 4: google-generativeai 未安装

**错误信息**:
```
ModuleNotFoundError: No module named 'google'
```

**解决方案**:
```bash
pip install google-generativeai
```

### 问题 5: 推理时间过长

**优化方案**:
```python
# 降低帧数
video_frames = load_video_frames(video_path, max_frames=8, target_fps=3)

# 减少输出长度
max_output_tokens=256  # 从 512 降到 256

# 使用更快的模型
runner = GeminiRunner("gemini-pro")  # 比 gemini-2.0-pro 更快
```

---

## 📊 批量处理示例

### 处理前 10 条数据

```python
from description_generation import load_qa_data
from benchmark.models.gemini_runner import GeminiRunner
import json
import os

# 初始化
data = load_qa_data("/path/to/csv")
runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

# 批量处理
results = []
for i, item in enumerate(data[:10]):
    print(f"处理 {i+1}/10...")
    
    prompt = f"""Based on:\n{item['facts_text']}\n\nGenerate a description."""
    
    try:
        # 直接调用 Gemini API
        model = runner.genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        results.append({
            'video_id': item['video_id'],
            'description': response.text
        })
    except Exception as e:
        print(f"  ❌ 失败: {e}")

# 保存结果
with open('results.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"完成！已保存 {len(results)} 条结果")
```

---

## ✅ 检查清单

- [ ] 获得 Gemini API 密钥
- [ ] 在 gemini_runner.py 中设置密钥
- [ ] 安装 google-generativeai
- [ ] （可选）配置代理或 VPN
- [ ] 测试单条数据推理
- [ ] 批量处理数据
- [ ] 保存和分析结果

---

## 📚 相关文件

- **主工作流**: `gemini_workflow.py`
- **Runner 代码**: `/home/24068286g/UString/VRU/src/benchmark/models/gemini_runner.py`
- **数据加载器**: `data_loader_csv.py`
- **视频处理**: `/home/24068286g/UString/VRU/src/api_test_framework/video_processor.py`

---

## 🔗 有用的链接

- Gemini API 文档: https://ai.google.dev/docs
- API 密钥管理: https://aistudio.google.com/apikey
- 模型列表: https://ai.google.dev/models
- Python 示例: https://github.com/google-gemini/generative-ai-python

---

## 💡 最佳实践

1. **安全管理密钥**
   - 不要提交到版本控制
   - 使用环境变量或 .env 文件
   - 定期轮换密钥

2. **错误处理**
   - 总是 try-except 包装 API 调用
   - 实现重试机制
   - 记录错误日志

3. **性能优化**
   - 调整 max_frames 和 target_fps
   - 使用合适的 temperature（0.1-0.3 用于确定性任务）
   - 批量处理而不是逐个请求

4. **成本控制**
   - 监控 API 使用量
   - 使用更小的模型（gemini-pro）开发
   - 设置合理的 max_output_tokens

---

*更新日期: 2026-01-18*
