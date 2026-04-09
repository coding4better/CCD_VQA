# Gemini API Complete Guide

## 1. Get a Gemini API key

### 1.1 Create the key
Go to: https://aistudio.google.com/apikey

Steps:
1. Sign in with a Google account
2. Click "Create API Key"
3. Select "Create new secret key"
4. Copy the generated key

### 1.2 Verify the key
```bash
curl https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY
```

---

## 2. Configure the key in code

### Option A: Edit `gemini_runner.py` (recommended)

Edit: `/home/24068286g/UString/VRU/src/benchmark/models/gemini_runner.py`

Find:
```python
self.api_key = "your_gemini_api_key_here"
```

Replace it with your real key:
```python
self.api_key = "AIza...your_actual_key..."
```

### Option B: Use an environment variable

```bash
export GEMINI_API_KEY="your_actual_api_key"
```

`GeminiRunner` will read it automatically.

### Option C: Set it dynamically in a script

```python
from benchmark.models.gemini_runner import GeminiRunner

runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_actual_api_key"
runner._init_api()
```

---

## 3. Run the inference workflow

### 3.1 Use the full workflow script

```bash
cd /home/24068286g/CCD_VQA/VRU/src/description_generation
python gemini_workflow.py
```

### 3.2 Quick test with one sample

```python
from description_generation import load_qa_data
from benchmark.models.gemini_runner import GeminiRunner

data = load_qa_data("/path/to/csv")
item = data[0]

runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

prompt = f"Based on:\n{item['facts_text']}\n\nGenerate a description."
result = runner.predict(item['video_id'], prompt, None)
print(result)
```

### 3.3 Full inference with video frames

```python
from description_generation import load_qa_data
from api_test_framework.video_processor import load_video_frames
from benchmark.models.gemini_runner import GeminiRunner

data = load_qa_data("/path/to/csv")
item = data[0]
video_frames = load_video_frames("/path/to/video.mp4", max_frames=16, target_fps=6)

runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

prompt = f"Based on:\n{item['facts_text']}\n\nGenerate a description."
result = runner.predict(item['video_id'], prompt, video_frames)
print(result)
```

---

## 4. Configuration tips

### Gemini model selection

```python
models = [
    "gemini-2.5-pro",   # newest and strongest
    "gemini-2.0-pro",   # stable and capable
    "gemini-pro",       # general-purpose
]

runner = GeminiRunner("gemini-2.5-pro")
```

### Inference parameter tuning

```python
response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=512,
        top_p=0.9,
        top_k=40,
    )
)
```

### Video frame parameter tuning

```python
video_frames = load_video_frames(
    video_path,
    max_frames=32,
    target_fps=10,
)
```

---

## 5. Common issues and fixes

### Issue 1: Region not supported

```text
Location not supported for the API use
```

Fix:
```bash
export HTTP_PROXY="http://proxy:port"
export HTTPS_PROXY="http://proxy:port"
```

Or use a VPN / run in a supported region.

### Issue 2: Invalid API key

```text
INVALID_ARGUMENT: API key not valid
Invalid API Key
```

Fix:
1. Check that the key was copied correctly
2. Check whether the key is enabled
3. Generate a new key

### Issue 3: Quota exhausted

```text
RESOURCE_EXHAUSTED: Quota exceeded
RATE_LIMIT: Please retry after
```

Fix:
1. Wait for the quota to reset
2. Upgrade to a paid plan
3. Reduce request frequency

### Issue 4: `google-generativeai` not installed

```text
ModuleNotFoundError: No module named 'google'
```

Fix:
```bash
pip install google-generativeai
```

### Issue 5: Inference is too slow

```python
video_frames = load_video_frames(video_path, max_frames=8, target_fps=3)
max_output_tokens = 256
runner = GeminiRunner("gemini-pro")
```

---

## 6. Batch processing example

### Process the first 10 samples

```python
from description_generation import load_qa_data
from benchmark.models.gemini_runner import GeminiRunner

data = load_qa_data("/path/to/csv")
runner = GeminiRunner("gemini-2.0-flash")
runner.api_key = "your_api_key"
runner._init_api()

results = []
for index, item in enumerate(data[:10]):
    print(f"Processing {index + 1}/10...")

    prompt = f"Based on:\n{item['facts_text']}\n\nGenerate a description."

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

- **Main workflow**: `gemini_workflow.py`
- **Runner code**: `/home/24068286g/UString/VRU/src/benchmark/models/gemini_runner.py`
- **Data loader**: `data_loader_csv.py`
- **Video processing**: `/home/24068286g/UString/VRU/src/api_test_framework/video_processor.py`

---

## Useful links

- Gemini API docs: https://ai.google.dev/docs
- API key management: https://aistudio.google.com/apikey
- Model list: https://ai.google.dev/models
- Python examples: https://github.com/google-gemini/generative-ai-python

---

## Best practices

1. **Keep the key secure**
    - Do not commit it to version control
    - Use environment variables or a `.env` file
    - Rotate keys regularly

2. **Handle errors properly**
    - Always wrap API calls in `try`/`except`
    - Implement retries
    - Log errors

3. **Optimize performance**
    - Tune `max_frames` and `target_fps`
    - Use a suitable `temperature` value (`0.1-0.3` for deterministic tasks)
    - Batch requests instead of sending them one by one

4. **Control cost**
    - Monitor API usage
    - Use a smaller model such as `gemini-pro` for development
    - Set a reasonable `max_output_tokens`

---

*Updated: 2026-01-18*
