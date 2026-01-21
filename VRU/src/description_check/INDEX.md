# 📑 文件导航和快速索引

## 🚀 我应该先读什么？

### 如果你只有 5 分钟 ⏱️
👉 **[QUICKSTART.md](./QUICKSTART.md)** - 快速启动指南

### 如果你有 15 分钟 ⏱️
👉 **[README.md](./README.md)** - 详细使用文档

### 如果你需要深入了解 ⏱️
👉 **[IMPLEMENTATION.md](./IMPLEMENTATION.md)** - 完整技术文档

### 如果你想看一览表 ⏱️
👉 **[PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)** - 项目交付总结

---

## 📂 文件详解

### 可执行文件

#### 1. `exp2_consistency_check.py` 🐍
**类型**: 完整 Python 脚本  
**大小**: 13 KB  
**用途**: 主程序，包含所有功能  

**包含的函数**:
- `load_baseline_descriptions()` - 加载 Baseline 数据
- `load_qa_data()` - 加载 QA 数据
- `extract_qa_sentences()` - 提取事实句子
- `check_consistency()` - LLM 一致性检查
- `evaluate_descriptions()` - 批量评估
- `generate_statistics()` - 统计分析
- `plot_consistency_boxplot()` - 绘制图表
- `main()` - 主程序入口

**运行方式**:
```bash
python exp2_consistency_check.py
```

**适用场景**:
- 本地 Linux/Mac/Windows 运行
- 需要完整自动化的项目
- 集成到其他 Python 项目

---

#### 2. `exp2_consistency_check.ipynb` 📔
**类型**: Jupyter Notebook  
**大小**: 119 B（创建时）+ 内容  
**用途**: 交互式版本，推荐用于 Google Colab  

**包含的 10 个部分**:
1. Section 1 - 安装依赖
2. Section 2 - 导入库和设置
3. Section 3 - 加载数据
4. Section 4 - 定义 LLM 检查器
5. Section 5 - 评估 Baseline
6. Section 6 - 评估 Refined
7. Section 7 - 统计分析
8. Section 8 - 绘制箱线图
9. Section 9 - 保存结果
10. Section 10 - 总结和建议

**运行方式**:
```bash
# 在 Google Colab 中打开
# 或本地运行
jupyter notebook exp2_consistency_check.ipynb
```

**优点**:
- 可逐个单元格运行
- 交互式调试
- 易于修改参数
- Colab 支持无 VPN 访问

---

#### 3. `verify_setup.py` ✅
**类型**: 验证脚本  
**大小**: 8.2 KB  
**用途**: 验证环境和依赖  

**检查项目**:
- Python 版本 (3.7+)
- 依赖包安装
- 数据文件存在
- API 密钥配置
- 输出目录权限
- 脚本文件完整性
- 文件访问权限

**运行方式**:
```bash
python verify_setup.py
```

**何时运行**:
- 第一次配置环境时
- 遇到问题时
- 切换环境时

---

#### 4. `usage_examples.py` 💡
**类型**: 示例代码  
**大小**: 7.9 KB  
**用途**: 6 个实际使用示例  

**包含的示例**:
1. **基础使用** - 导入和调用函数
2. **批量评估** - 评估多个视频
3. **自定义评估** - 自己的评估逻辑
4. **结果分析** - 分析现有结果
5. **自定义可视化** - 自己的图表
6. **完整工作流** - 端到端流程

**运行方式**:
```bash
python usage_examples.py [1-6]  # 运行指定示例
python usage_examples.py        # 运行所有示例
```

**何时使用**:
- 学习如何使用库
- 需要自定义功能时
- 集成到自己的项目中

---

### 文档文件

#### 1. `README.md` 📖 (主文档)
**大小**: 7.4 KB  
**适合**: 详细学习

**包含章节**:
- 项目概述
- 使用方法 (Python + Colab)
- 配置参数
- 输出文件说明
- 评估逻辑细节
- 论文中的使用
- 常见问题 (10+)
- 扩展建议

**阅读时间**: 15-20 分钟

---

#### 2. `QUICKSTART.md` 🚀 (快速指南)
**大小**: 2.9 KB  
**适合**: 快速上手

**包含**:
- 3 分钟快速开始
- 前置准备清单
- 4 个快速步骤
- 预期输出
- 常见问题 (3+)
- 下一步建议

**阅读时间**: 5 分钟

---

#### 3. `IMPLEMENTATION.md` 🔧 (技术深度)
**大小**: 12 KB  
**适合**: 深入学习

**包含章节**:
- 核心算法
- 代码结构详解
- 参数说明
- 输出文件格式
- 自定义扩展 (4+)
- 常见陷阱 (4+)
- 测试方法
- 论文使用示例

**阅读时间**: 20-30 分钟

---

#### 4. `PROJECT_SUMMARY.md` 📋 (交付总结)
**大小**: 9 KB  
**适合**: 了解全景

**包含**:
- 已完成工作清单
- 技术规格
- 输出示例
- 文档内容总结
- 项目统计
- 项目亮点

**阅读时间**: 10 分钟

---

#### 5. `__init__.py` 🐍 (包初始化)
**大小**: 383 B  
**用途**: 使 description_check 成为可导入的包

**内容**:
```python
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"
```

---

### 辅助文件

#### `INDEX.md` (本文件)
**用途**: 文件导航和快速索引  
**阅读时间**: 5 分钟

---

## 🎯 按任务快速导航

### 任务: "我想快速运行一次实验"
1. 阅读: [QUICKSTART.md](./QUICKSTART.md)
2. 运行: `python verify_setup.py`
3. 执行: `jupyter notebook exp2_consistency_check.ipynb`
4. 查看: `results/fig1_consistency.png`

### 任务: "我想深入理解代码"
1. 阅读: [IMPLEMENTATION.md](./IMPLEMENTATION.md)
2. 学习: [usage_examples.py](./usage_examples.py)
3. 查看: `exp2_consistency_check.py`
4. 实践: 运行示例代码

### 任务: "我想自定义参数"
1. 查看: [README.md](./README.md) - 配置参数部分
2. 编辑: `exp2_consistency_check.py` 或 `.ipynb`
3. 验证: `python verify_setup.py`
4. 运行: 执行修改后的脚本

### 任务: "我想在论文中使用结果"
1. 生成: 运行实验生成 `fig1_consistency.png`
2. 使用: 直接插入论文
3. 引用: 参考 [README.md](./README.md) 的论文使用部分
4. 解释: 使用统计数据和报告内容

### 任务: "我遇到了问题"
1. 验证: `python verify_setup.py`
2. 检查: [README.md](./README.md) - 常见问题部分
3. 查看: [IMPLEMENTATION.md](./IMPLEMENTATION.md) - 常见陷阱部分
4. 查询: 日志输出或错误信息

### 任务: "我想扩展功能"
1. 学习: [usage_examples.py](./usage_examples.py) 的示例 3-5
2. 参考: [IMPLEMENTATION.md](./IMPLEMENTATION.md) - 自定义扩展部分
3. 编码: 添加自己的函数
4. 测试: 使用示例代码测试

---

## 📚 文档关系图

```
├─ 快速入门
│  └─ QUICKSTART.md (5 min)
│
├─ 详细使用
│  ├─ README.md (15 min)
│  └─ usage_examples.py (10 min)
│
├─ 技术深度
│  └─ IMPLEMENTATION.md (20 min)
│
├─ 项目概览
│  └─ PROJECT_SUMMARY.md (10 min)
│
└─ 可执行代码
   ├─ exp2_consistency_check.py (完整实现)
   ├─ exp2_consistency_check.ipynb (交互式)
   ├─ verify_setup.py (环境验证)
   └─ usage_examples.py (6 个示例)
```

---

## 🔑 关键概念快速查询

### Baseline 是什么？
👉 [README.md - 评估逻辑](./README.md#评估逻辑-nli-evaluator)  
👉 [IMPLEMENTATION.md - 核心算法](./IMPLEMENTATION.md#-核心算法)

### LLM 检查器怎么工作？
👉 [README.md - 裁判逻辑](./README.md#裁判逻辑)  
👉 [IMPLEMENTATION.md - 步骤3](./IMPLEMENTATION.md#步骤-3-llm-一致性检查)

### 如何修改采样大小？
👉 [README.md - 配置参数](./README.md#修改采样大小)  
👉 [usage_examples.py - 示例](./usage_examples.py)

### 箱线图什么时候生成？
👉 [IMPLEMENTATION.md - 步骤6](./IMPLEMENTATION.md#步骤-6-可视化)  
👉 [README.md - 输出文件](./README.md#-箱线图-)

### 如何在论文中使用？
👉 [README.md - 论文中的使用](./README.md#论文中的使用)  
👉 [PROJECT_SUMMARY.md - 应用部分](./PROJECT_SUMMARY.md#-在论文中的应用)

---

## 💬 常见问题速查

| 问题 | 文档位置 |
|------|--------|
| "哪个脚本应该运行？" | QUICKSTART.md |
| "我的依赖没装好" | verify_setup.py 运行 |
| "API 密钥错了" | README.md 常见问题 Q1 |
| "文件找不到" | README.md 常见问题 Q3 |
| "结果格式是什么？" | IMPLEMENTATION.md 输出文件说明 |
| "如何修改参数？" | README.md 配置参数 |
| "想添加自定义功能" | usage_examples.py 示例 3 |
| "怎样在论文里用？" | README.md 论文中的使用 |

---

## 📞 快速联系地图

```
遇到问题？
  ├─ 环境问题 → verify_setup.py
  ├─ 使用问题 → QUICKSTART.md / README.md
  ├─ 技术问题 → IMPLEMENTATION.md
  ├─ 代码问题 → usage_examples.py
  └─ 一般问题 → PROJECT_SUMMARY.md
```

---

## 🎓 学习路径

### 初级 (第 1 天)
- [ ] 阅读 QUICKSTART.md (5 min)
- [ ] 运行 verify_setup.py (2 min)
- [ ] 运行 Notebook 的前 5 个部分 (10 min)
- [ ] 查看生成的图表 (2 min)

### 中级 (第 2-3 天)
- [ ] 完整阅读 README.md (20 min)
- [ ] 运行 usage_examples.py 的示例 1-3 (20 min)
- [ ] 修改参数重新运行 (15 min)
- [ ] 分析输出结果 (20 min)

### 高级 (第 4-7 天)
- [ ] 深入学习 IMPLEMENTATION.md (30 min)
- [ ] 阅读完整的 exp2_consistency_check.py (30 min)
- [ ] 运行示例 4-6 (30 min)
- [ ] 实现自定义功能 (60+ min)

### 应用 (第 8+ 天)
- [ ] 扩大采样规模
- [ ] 加载真实的 Refined 数据
- [ ] 在论文中使用结果
- [ ] 进行统计显著性检验

---

## 📊 文件树

```
description_check/
├── 📄 exp2_consistency_check.py          ← 主程序
├── 📔 exp2_consistency_check.ipynb       ← Colab 版本
├── 🔍 verify_setup.py                    ← 环境检查
├── 💡 usage_examples.py                  ← 示例代码
├── 📖 README.md                          ← 详细文档
├── 🚀 QUICKSTART.md                      ← 快速指南
├── 🔧 IMPLEMENTATION.md                  ← 技术文档
├── 📋 PROJECT_SUMMARY.md                 ← 项目总结
├── 📑 INDEX.md                           ← 本文件
├── 🐍 __init__.py                        ← 包初始化
└── 📁 results/                           ← 输出目录
    ├── fig1_consistency.png              ← 论文图表
    ├── consistency_evaluation_*.json     ← 详细数据
    ├── consistency_scores_*.csv          ← CSV 数据
    └── consistency_report_*.txt          ← 文本报告
```

---

## 🎯 推荐阅读顺序

### 方案 A: 快速上手 (20 分钟)
1. QUICKSTART.md
2. 运行 verify_setup.py
3. 运行 exp2_consistency_check.py 或 .ipynb
4. 查看结果

### 方案 B: 全面学习 (1 小时)
1. QUICKSTART.md
2. README.md
3. usage_examples.py
4. 运行实验
5. 分析结果

### 方案 C: 深度掌握 (2-3 小时)
1. PROJECT_SUMMARY.md
2. QUICKSTART.md
3. README.md
4. IMPLEMENTATION.md
5. exp2_consistency_check.py (代码)
6. usage_examples.py (示例)
7. 实践和定制

---

## ✨ 最后提示

- 📌 **第一次运行**: 使用 QUICKSTART.md
- 🔧 **遇到问题**: 运行 verify_setup.py
- 📖 **深入学习**: 阅读 IMPLEMENTATION.md
- 💻 **看代码**: 查看 usage_examples.py
- 🎓 **论文使用**: 参考 README.md 论文部分
- 🚀 **准备好了**: 开始你的实验！

---

**选择你的下一步**:
- 👉 [快速启动 (5 min)](./QUICKSTART.md)
- 👉 [详细文档 (20 min)](./README.md)
- 👉 [技术深度 (30 min)](./IMPLEMENTATION.md)
- 👉 [项目总结](./PROJECT_SUMMARY.md)

---

**最后更新**: 2025-01-19  
**文档版本**: 1.0  
**状态**: 完成 ✅
