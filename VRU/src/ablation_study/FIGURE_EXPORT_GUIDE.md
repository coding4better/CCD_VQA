# VQA Dataset Quality Analysis - Figure Export Guide

## Document Location
📄 Main Report: `e:\study\project\CCD_VQA\VRU\VQA_Dataset_Quality_Analysis_Report.md`

---

## Required Figures for Paper

The main report references 4 key figures. Here's how to export them from the Jupyter notebook:

### Figure 1: `bert_quality_analysis_boxplot.png`
**Cell:** `plot_quality_comparison()` function execution  
**Metric:** Box plots of quality metrics (Option Diversity, Distractor Quality, Question Relevance, Separability)  
**Size:** Save as 150 DPI, 14×10 inches

**In Notebook:**
1. Run the cell containing `plot_quality_comparison(combined_results)`
2. Right-click on the displayed plot
3. Select "Save image as..." → name it `bert_quality_analysis_boxplot.png`

---

### Figure 2: `bert_quality_analysis_kde.png`
**Cell:** `plot_distributions()` function execution  
**Metric:** Kernel Density Estimation distributions for each metric  
**Size:** Save as 150 DPI, 14×10 inches

**In Notebook:**
1. Run the cell containing `plot_distributions(combined_results)`
2. Save the output image as `bert_quality_analysis_kde.png`

---

### Figure 3: `bert_quality_analysis_radar.png`
**Cell:** `plot_radar_comparison()` function execution  
**Metric:** Radar chart comparing all 4 configurations across metrics  
**Size:** Save as 150 DPI, 8×8 inches

**In Notebook:**
1. Run the cell containing `plot_radar_comparison(summary_stats)`
2. Save the output image as `bert_quality_analysis_radar.png`

---

### Figure 4: `bert_quality_analysis_final.png`
**Cell:** `plot_final_comparison()` function execution  
**Metric:** Composite score bar chart + individual metrics grouped comparison  
**Size:** Save as 150 DPI, 14×5 inches (horizontal layout)

**In Notebook:**
1. Run the cell containing `plot_final_comparison(combined_results, summary_stats)`
2. Save the output image as `bert_quality_analysis_final.png`

---

## File Organization for Paper Submission

Recommended folder structure:
```
project_paper/
├── VQA_Dataset_Quality_Analysis_Report.md          ← Main document
├── figures/
│   ├── bert_quality_analysis_boxplot.png           ← Figure 1
│   ├── bert_quality_analysis_kde.png               ← Figure 2
│   ├── bert_quality_analysis_radar.png             ← Figure 3
│   └── bert_quality_analysis_final.png             ← Figure 4
└── data/
    ├── summary_statistics.csv                      ← From notebook Part 15
    └── detailed_results.csv                        ← Full analysis results
```

---

## Notebook Cells for Data Export

### Export Summary Statistics
**Cell:** Contains `summary_stats.to_csv()`
```python
# Automatically executed in notebook Part 15
# Output location varies; by default saved to Google Drive
```

### Export Detailed Results
**Cell:** Contains `combined_results.to_csv()`
```python
# Automatic in notebook Part 15
detail_path = DATA_DIR / f"bert_analysis_detailed_{timestamp}.csv"
```

### Generate Report TXT
**Cell:** Contains report generation code
```python
# Creates comprehensive text report (see Part 15)
report_path = DATA_DIR / f"bert_analysis_report_{timestamp}.txt"
```

---

## Figure Customization for Paper Target

### For IEEE/ACM Templates
- **Format:** PDF or high-resolution PNG (300+ DPI)
- **Size:** Column width 3.5" (8.9cm) or page width 7" (17.8cm)
- **Font:** Use serif fonts (Times New Roman 10pt) to match paper text

**Recommended changes to code:**
```python
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
# Save at 300 DPI instead of 150
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight', format='pdf')
```

### For arXiv/TechReport
- **Format:** PNG or PDF
- **DPI:** 150 minimum (300 preferred)
- **Size:** 6-8 inches wide recommended

---

## Critical Data Points for Text

Copy-paste these key numbers into your paper:

### Summary Table (Table 1)
```
Composite Quality Scores (Mean ± Std):
- 2 Options:  0.4405
- 3 Options:  0.4523
- 4 Options:  0.4638 ← BEST
- 5 Options:  0.4612
```

### Statistical Significance
```
Kruskal-Wallis H-test (all metrics):
p-value < 0.001 ✅ Highly Significant

4-Options vs 2-Options (Separability):
Effect Size = 0.289 → LARGE practical significance
p < 0.001 → Statistically significant
```

### Improvement Metrics
```
4-Options vs 2-Options:
- Separability: +11.9% improvement (0.589 → 0.659)
- Diversity: +29.8% improvement (0.312 → 0.405)
- Composite Score: +5.3% improvement
```

---

## Integration with Main Paper Structure

### Suggested Section Placement

**Section 4: Experiments & Analysis**
1. Introduce BERT-based quality metrics
2. Display Table 1 (individual metrics comparison)
3. Include Figure 3 (radar chart) for visual overview
4. Report statistical significance (Kruskal-Wallis results)

**In Appendix or Supplementary Material**
1. Detailed metric definitions (Appendix A)
2. Statistical test details (Appendix B)
3. All 4 figures with extended captions
4. CSV data files for reproducibility

---

## Reproducibility & Citation

To enable reproducibility:

1. **Include notebook URL/repository link** if sharing publicly
2. **Specify model version:** Sentence-BERT `all-MiniLM-L6-v2`
3. **Include random seed** (if any randomization in sampling)
4. **Link exact CSV data files** used in analysis
5. **Provide Python environment requirements:**
   ```
   sentence-transformers==2.2.2
   pandas==1.5.3
   numpy==1.24.0
   scipy==1.10.0
   matplotlib==3.7.0
   seaborn==0.12.0
   ```

---

## Verification Checklist Before Submission

- [ ] All 4 figure images present at correct DPI (300+)
- [ ] Figure captions match report exactly
- [ ] Table 1 numbers copied correctly to paper text
- [ ] Statistical significance statements accurate
- [ ] Notebook execution validated (no errors)
- [ ] Data CSV files archived for supplementary materials
- [ ] Code cell references in text match notebook structure
- [ ] Markdown report converts cleanly to PDF/LaTeX

---

## Quick Export Script (Optional)

Save this Python snippet to automate figure export:

```python
# save_figures.py
import matplotlib.pyplot as plt
from pathlib import Path

# Directory to save figures
OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# High DPI for paper submission
DPI = 300

# After running analysis in notebook:
# Copy current figures
figures = plt.get_fignums()
for i, fig_num in enumerate(figures):
    fig = plt.figure(fig_num)
    fig.savefig(OUTPUT_DIR / f"bert_quality_analysis_{i}.png", 
                dpi=DPI, bbox_inches='tight')
    print(f"✅ Figure {i} saved")
```

---

## Support & Questions

For any clarification on:
- Metric interpretation → See Report Section 2.2
- Statistical details → See Appendix B
- Figure generation → See corresponding notebook cells
- Data format → Inspect CSV headers

