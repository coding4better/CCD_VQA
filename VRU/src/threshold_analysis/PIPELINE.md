# 阈值确定流程管道（Pipeline）

## 🎯 整体目标
从 1500 个视频中，通过科学的指标筛选，最终确定用于视频过滤的最优阈值组合。

---

## 📋 流程分解：4 个阶段

### 阶段 1️⃣：确定指标（Metric Determination）
**目的**：为每个视频计算两个关键指标

**输入**：
- CCD 原始视频和标注（1500个视频）
- YOLO 检测结果（NPZ 格式，含目标框、置信度、特征）

**处理**：
```
对每个视频：
  1. 加载事故标注，提取事故前后 ±30 帧的窗口
  2. 对高置信度检测(>0.5)取帧内特征平均
  3. 计算相邻帧特征的欧氏距离，取最大值
  4. 用全局 P95 距离归一化 → Dynamic Change 分数
  5. 统计窗口内最多有多少个高置信度检测对象 → Scene Complexity
```

**输出文件**：
- `threshold_analysis/00_raw_metrics.csv` — 1500×5 表格（video_name, accident_frame, dynamic_change, scene_complexity, window_length）

**负责脚本**：
- `threshold_analysis.py` — 主计算脚本
  - `compute_all_metrics()` — 两遍扫描（全局参考值 + 归一化指标）
  - `analyze_distribution()` — 分布统计
  - `suggest_thresholds()` — 基于分位数的初步建议

**输出**：
- `threshold_analysis/01_distribution_analysis.json` — 分布统计（min/max/mean/std/quantiles）
- `threshold_analysis/02_threshold_suggestions.json` — 分位数建议（P80、P70、P60 等）

---

### 阶段 2️⃣：筛选扫描（Threshold Sweep）
**目的**：理解不同阈值组合下有多少个视频通过筛选

**输入**：
- `threshold_analysis/00_raw_metrics.csv`

**处理**：
```
对每个 (Complexity, Dynamic) 组合：
  应用 AND 逻辑筛选：
    mask = (complexity >= C_th) AND (dynamic >= D_th)
  统计通过的视频数、保留率、质量提升
```

**输出文件**：
- `threshold_analysis/03_threshold_sweep_table.csv` — 完整扫描表（所有组合的结果）
- `threshold_analysis/04_candidate_thresholds.json` — 候选方案列表（按样本量约束筛选）

**负责脚本**：
- `threshold_sweep.py`
  - `threshold_sweep()` — 网格扫描
  - `identify_candidates()` — 基于样本量约束找候选方案

---

### 阶段 3️⃣：确定最优阈值（Threshold Selection）
**目的**：根据业务需求（样本量、质量、复现性），选择最终阈值

**输入**：
- `threshold_analysis/04_candidate_thresholds.json`

**决策维度**：
```
1. 样本量约束：最少需要多少个？(通常 ≥150-200)
2. 质量需求：指标提升比例是否满足？
3. 复现性：阈值是否是标准分位数？(如 P70, P60)
4. 平衡度：两个维度是否同时改善？
```

**输出**：
- `threshold_analysis/05_final_decision.json` — 最终选定的阈值 + 决策理由
  ```json
  {
    "selected_thresholds": {
      "complexity": 6,
      "dynamic": 0.7306
    },
    "logic": "AND",
    "expected_count": 178,
    "expected_percentage": 11.9,
    "rationale": "...",
    "alternatives": [...]
  }
  ```

**决策工具**：
- 人工或脚本（待实现）
  - 基于 `04_candidate_thresholds.json` 的条件筛选
  - 生成决策报告

---

### 阶段 4️⃣：生成最终列表（Final List Generation）
**目的**：应用最终阈值，导出通过筛选的视频列表

**输入**：
- `threshold_analysis/00_raw_metrics.csv`
- `threshold_analysis/05_final_decision.json`

**处理**：
```
mask = (metrics['scene_complexity'] >= selected_complexity) 
     & (metrics['dynamic_change'] >= selected_dynamic)
final_videos = metrics[mask][['video_name', 'scene_complexity', 'dynamic_change', ...]]
```

**输出文件**：
- `threshold_analysis/06_final_filtered_videos.json` — 最终通过筛选的视频列表
  ```json
  {
    "description": "最终筛选结果",
    "thresholds": {"complexity": 6, "dynamic": 0.7306},
    "total_count": 178,
    "videos": [
      {"video_name": "000001.mp4", "scene_complexity": 8, "dynamic_change": 0.78, ...},
      ...
    ]
  }
  ```

**负责脚本**：
- `final_list_generator.py`（待实现）
  - 读 `05_final_decision.json`
  - 应用阈值
  - 导出 JSON/CSV

---

## 📁 文件分类表

### ✅ 必留（流程中核心步骤）

| 文件 | 阶段 | 用途 | 输入 | 输出 |
|------|------|------|------|------|
| `threshold_analysis.py` | 1 | 计算指标、分布、建议 | NPZ + 标注 | 00/01/02 |
| `threshold_sweep.py` | 2 | 网格扫描、候选方案 | 00_raw_metrics.csv | 03/04 |
| (新) `final_decision.py` | 3 | 决策逻辑、理由说明 | 04_candidates.json | 05_decision.json |
| (新) `final_list_generator.py` | 4 | 应用阈值、导出列表 | 00_raw_metrics + 05_decision | 06_final_videos |

### ⚠️ 需评估

| 文件 | 当前状态 | 是否在流程中 | 建议 |
|------|--------|----------|------|
| `threshold_determination.py` | 多种方法比较 | 否（流程已用 threshold_sweep.py） | 可删除（功能重复） |
| `threshold_exploration_unsupervised.py` | 无监督分析 | 否 | 需验证是否被引用；未使用可删除 |
| `EXPERIMENT_RESULTS.md` | 旧版本文档 | 否 | 更新或删除（信息可能过时） |
| `THRESHOLD_STRATEGY.md` | 策略说明 | 可能 | 可作为背景参考保留，或整合入 PIPELINE.md |
| `current_optimal_config.json` | 旧配置 | 否 | 应由新流程生成，删除 |
| `threshold_methods_comparison.json` | 方法对比 | 否 | 如无用可删除 |
| `unsupervised_*.{csv,json}` | 旧数据 | 否 | 删除（已由 threshold_analysis.py 替代） |

### 📂 输出文件（threshold_analysis/ 目录）

| 文件 | 阶段 | 说明 |
|------|------|------|
| `00_raw_metrics.csv` | 1 | 原始指标 |
| `01_distribution_analysis.json` | 1 | 分布统计 |
| `02_threshold_suggestions.json` | 1 | 初步分位数建议 |
| `03_threshold_sweep_table.csv` | 2 | 完整扫描表 |
| `04_candidate_thresholds.json` | 2 | 候选方案 |
| `05_final_decision.json` | 3 | 最终决策（待生成） |
| `06_final_filtered_videos.json` | 4 | 最终列表（待生成） |

---

## 🔄 执行流程

```bash
# 1. 阶段1：计算指标
cd /home/24068286g/CCD_VQA/VRU/src/threshold_analysis
python3 threshold_analysis.py
# 输出：00_raw_metrics.csv, 01_distribution.json, 02_suggestions.json

# 2. 阶段2：扫描阈值
python3 threshold_sweep.py
# 输出：03_threshold_sweep_table.csv, 04_candidates.json

# 3. 阶段3：手动决策或脚本决策
# 查看 04_candidates.json，选择最优方案
# 或运行 final_decision.py 自动决策
# 输出：05_final_decision.json

# 4. 阶段4：生成最终列表
python3 final_list_generator.py
# 输出：06_final_filtered_videos.json / .csv
```

---

## 📊 关键决策点

在阶段 3 选择最优阈值时，需要考虑：

1. **样本量**（阶段2的输出）
   - 过少（<150）：统计不稳定
   - 过多（>500）：质量可能下降

2. **质量指标**
   - Complexity 提升：相对基线的百分比
   - Dynamic 提升：相对基线的百分比

3. **分位数友好性**
   - 优先选 P50/P60/P70/P80 等标准分位数
   - 便于复现和解释

4. **业务约束**
   - 最终列表大小
   - 计算资源
   - 标注工作量

---

## 🎓 现状总结

| 阶段 | 完成度 | 备注 |
|------|--------|------|
| 1️⃣ 指标确定 | ✅ 100% | threshold_analysis.py 完成 |
| 2️⃣ 阈值扫描 | ✅ 100% | threshold_sweep.py 完成 |
| 3️⃣ 最优决策 | ⏳ 0% | 待实现决策脚本 |
| 4️⃣ 列表生成 | ⏳ 0% | 待实现列表生成脚本 |

下一步：实现阶段 3、4 的脚本，完成整个流程。
