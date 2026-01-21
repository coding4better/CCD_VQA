# 快速启动指南 (Quick Start)

## 🎯 3 分钟快速开始

### 前置准备
1. 获取 Gemini API 密钥：https://aistudio.google.com/apikey
2. 有访问数据文件的权限

### 步骤 1: 设置环境变量 (30 秒)

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 步骤 2: 在 Google Colab 上运行 (推荐)

**方式 A: 从 GitHub 或本地加载**
```python
# 在 Colab 中运行以下代码
!git clone <repo_url> /content/project
%cd /content/project/CCD_VQA/VRU/src/description_check
!jupyter nbconvert --to notebook --execute exp2_consistency_check.ipynb
```

**方式 B: 直接在 Colab 中编辑**
1. 打开 Colab 新建 notebook
2. 复制 `exp2_consistency_check.ipynb` 中的所有单元格
3. 修改数据路径为 Colab 路径
4. 执行

### 步骤 3: 本地运行 (可选)

```bash
# 进入目录
cd /home/24068286g/CCD_VQA/VRU/src/description_check

# 运行脚本
python exp2_consistency_check.py
```

### 步骤 4: 查看结果 (30 秒)

```bash
# 查看生成的文件
ls -lah results/

# 查看图表
open results/fig1_consistency.png  # macOS
xdg-open results/fig1_consistency.png  # Linux
```

## 📊 预期输出

运行完成后，你应该看到：

```
results/
├── fig1_consistency.png              # 论文用图表
├── consistency_evaluation_*.json     # 详细数据
├── consistency_scores_*.csv          # CSV 数据
└── consistency_report_*.txt          # 文本报告
```

## 🔧 如果遇到问题

### 问题 1: API 密钥无效
```bash
# 检查密钥
echo $GEMINI_API_KEY

# 重新设置
export GEMINI_API_KEY="new_key_here"
```

### 问题 2: 文件找不到
编辑脚本中的路径：
```python
BASELINE_DESC_PATH = "/your/path/to/gemini_descriptions_*.json"
QA_DATA_PATH = "/your/path/to/generated_vqa_eng.json"
```

### 问题 3: 速率限制
增加延迟（在脚本中找到）：
```python
time.sleep(0.5)  # 改为 1.0
```

## 💡 关键结果解释

### 箱线图说明
- **左边箱 (Baseline)**: 直接生成的描述质量
- **右边箱 (Refined)**: 改进方法的描述质量
- **如果右边的箱更高，说明改进方法有效**

### 数字解释
```
平均分: 0.75 意味着
- 75% 的描述与事实一致
- 25% 有矛盾或缺信息
```

## 🚀 下一步

1. ✅ 基础运行成功后，增加采样大小：
   ```python
   sample_size = min(50, len(common_video_ids))  # 从 5 改为 50
   ```

2. 📊 加载真实的 Refined 描述：
   - 在 Section 6 中修改代码
   - 替换临时的演示数据

3. 📈 添加统计显著性检验：
   ```python
   from scipy.stats import ttest_ind
   t, p = ttest_ind(baseline_scores, refined_scores)
   print(f"p-value: {p}")  # p < 0.05 表示显著差异
   ```

4. 🎓 在论文中使用结果

## 📚 完整文档

详见 [README.md](./README.md)

---

**需要帮助？** 检查日志输出的错误信息，或参考 README 的常见问题部分。
