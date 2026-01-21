# 描述一致性验证实验 (Exp2)

## 📋 概述

验证生成的视频描述与原始 QA 事实的一致性，用于论文的 **Motivation** 部分，证明：
- **不加控制的描述容易出错**（Baseline 一致性分数较低）
- **改进方法的准确率更高**（Refined 一致性分数明显提升）

## 🎯 实验目标

使用 LLM 作为"逻辑检查器"，评估：
1. **Baseline 描述**（直接由 Gemini API 生成）
2. **Refined 描述**（经过改进方法处理）

与原始 QA 数据中 6 个验证事实的一致性。

## 📂 文件结构

```
/home/24068286g/CCD_VQA/VRU/src/description_check/
├── exp2_consistency_check.py          # Python 脚本版本
├── exp2_consistency_check.ipynb       # Jupyter Notebook 版本（推荐用于 Colab）
├── README.md                          # 本文件
└── results/
    ├── fig1_consistency.png           # 箱线图（论文中使用）
    ├── consistency_evaluation_*.json   # 详细评估数据
    ├── consistency_scores_*.csv        # CSV 格式数据
    └── consistency_report_*.txt        # 文本摘要报告
```

## 🚀 使用方法

### 方法 1: Google Colab (推荐)

1. **打开 Notebook**
   - 在 Google Colab 中打开 `exp2_consistency_check.ipynb`
   - 或使用链接: [在 Colab 中打开](#)

2. **设置 API 密钥**
   ```python
   # 在第 2 个单元格中替换为你的 API 密钥
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

3. **运行单元格**
   - 从第 1 个单元格开始逐个运行
   - 或直接点击"全部运行"

4. **下载结果**
   - 图表：`fig1_consistency.png`
   - 数据：JSON/CSV 文件

### 方法 2: 本地 Python 脚本

1. **设置环境**
   ```bash
   cd /home/24068286g/CCD_VQA/VRU/src/description_check
   
   # 安装依赖
   pip install google-generativeai pandas numpy matplotlib tqdm
   
   # 设置 API 密钥
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

2. **运行脚本**
   ```bash
   python exp2_consistency_check.py
   ```

3. **查看结果**
   ```bash
   ls -la results/
   ```

## 🔧 配置参数

### 数据路径（可在脚本中修改）

```python
BASELINE_DESC_PATH = "/path/to/gemini_descriptions_*.json"
QA_DATA_PATH = "/path/to/generated_vqa_eng.json"
OUTPUT_DIR = "/path/to/results/"
```

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_size` | 5 | 评估的视频数量（演示用）|
| `model_name` | `gemini-2.0-flash` | 评估使用的 LLM 模型 |
| `temperature` | 0.1 | 模型温度（越低越确定） |
| `timeout` | 5.0 | API 请求超时时间（秒） |

### 修改采样大小

**Notebook 中**：找到以下行并修改
```python
sample_size = min(10, len(common_video_ids))  # 改为你需要的数量
```

**Python 脚本中**：修改函数调用
```python
baseline_scores, refined_scores = evaluate_descriptions(
    baseline_descriptions,
    qa_data,
    api_key,
    sample_size=50  # 改为实际需要的数量
)
```

## 📊 评估逻辑

### 系统 Prompt
```
"You are a logic checker. Determine if the Description entails the Verified Fact."
```

### 用户 Prompt 模板
```
Description: {video_description}

Verified Fact: {qa_sentence}

Output 1 if consistent, 0 if contradictory or missing key info. Only output the number.
```

### 评分规则
- **1**: 描述与事实一致，包含所有关键信息
- **0**: 描述与事实矛盾，或缺少关键信息

### 计算方式

对于每个视频：
1. 提取 6 个 QA 对应的事实句子
2. 对每个事实，调用 LLM 获得一致性评分 (0 或 1)
3. 计算平均分：`avg_score = sum(scores) / 6`

## 📈 输出文件说明

### 1. 箱线图 (`fig1_consistency.png`)
- **用途**：在论文中展示 Baseline vs Refined 的对比
- **内容**：
  - 左箱：Baseline 描述的一致性分数分布
  - 右箱：Refined 描述的一致性分数分布
  - 菱形符号：平均值
  - 蓝线：中位数

### 2. JSON 数据 (`consistency_evaluation_*.json`)
```json
{
  "timestamp": "20250119_120000",
  "baseline": {
    "scores": [0.75, 0.83, 0.67, ...],
    "statistics": {
      "mean": 0.75,
      "std": 0.08,
      ...
    }
  },
  "refined": { ... },
  "comparison": {
    "improvement_percent": 12.5,
    "absolute_improvement": 0.09
  }
}
```

### 3. CSV 数据 (`consistency_scores_*.csv`)
```
video_id,baseline_score,refined_score
3,0.75,0.83
18,0.67,0.75
...
```

### 4. 文本报告 (`consistency_report_*.txt`)
包含完整的统计数据和分析结果，可直接粘贴到论文中。

## 💡 关键步骤

### 步骤 1: 加载数据 ✅
- Baseline 描述：从 Gemini API 结果文件加载
- QA 数据：从原始 JSON 文件加载

### 步骤 2: 提取事实 ✅
- 从每个 VQA 对象提取 (问题, 答案) 对
- 组合为"事实句子"

### 步骤 3: LLM 评估 ✅
- 对每个 (描述, 事实) 对调用 Gemini API
- 获取 1 或 0 的一致性分数

### 步骤 4: 统计分析 ✅
- 计算平均分、标准差、中位数等
- 对比 Baseline 和 Refined 的统计指标

### 步骤 5: 可视化 ✅
- 绘制箱线图
- 添加统计信息和图例

### 步骤 6: 保存结果 ✅
- 保存图表为 PNG
- 保存详细数据为 JSON/CSV
- 生成文本摘要报告

## ⚠️ 常见问题

### Q1: API 密钥错误
**A**: 确保：
1. 密钥从 [Google AI Studio](https://aistudio.google.com/apikey) 获取
2. 已正确设置环境变量或在代码中替换
3. 密钥未过期

### Q2: 速率限制
**A**: 
- 增加请求间延迟：修改 `time.sleep()` 的值
- 降低采样大小
- 使用 Colab Pro 获得更高的限制

### Q3: QA 数据加载失败
**A**: 
- 检查文件路径是否正确
- 确保 JSON 文件格式有效
- 验证数据文件不为空

### Q4: 如何加载真实的 Refined 描述？
**A**: 在 Section 6 中，替换以下代码：
```python
# 原来
refined_desc = baseline_desc  # 临时方案

# 改为
refined_desc = refined_descriptions[video_id]['description']
```

并在 Section 3 后添加加载 Refined 数据的函数。

## 📚 论文中的使用

### Figure 建议标题
```
Figure X: Consistency Score Distribution Comparison
(a) Baseline descriptions generated by Gemini API
(b) Refined descriptions with our method
```

### 文本说明
```
We evaluated the consistency of both baseline and refined descriptions 
against verified facts using an LLM-based logic checker. As shown in 
Figure X, refined descriptions demonstrate significantly higher 
consistency scores (mean ± std: X.XX ± 0.XX) compared to baseline 
descriptions (X.XX ± 0.XX), with a relative improvement of X%.
```

## 🔬 实验扩展建议

1. **增加采样量**
   - 当前：5 个视频（演示）
   - 建议：50-100 个视频（充分评估）

2. **统计显著性检验**
   ```python
   from scipy.stats import ttest_ind
   t_stat, p_value = ttest_ind(baseline_scores, refined_scores)
   ```

3. **按视频属性分组分析**
   - 按碰撞类型
   - 按天气条件
   - 按道路类型

4. **错误分析**
   - 统计哪些事实最容易出现不一致
   - 分析失败的模式

5. **细粒度评估**
   - 不只计算 0/1，还可使用相似度分数
   - 使用更复杂的评估 Prompt

## 📞 支持

如有问题，请：
1. 检查日志输出信息
2. 查看生成的报告文件
3. 参考常见问题部分
4. 检查 API 文档

## 📝 许可证

此脚本为研究项目的一部分，遵循项目主许可证。

---

**最后更新**: 2025-01-19  
**版本**: 1.0
