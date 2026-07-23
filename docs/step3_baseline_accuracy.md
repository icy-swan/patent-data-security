# Step 1 / Step 2 准确率复算（基于 Step 3 人工金标准）

- 金标准来源：`data/step3/result.csv`（`result_positive.csv` + `result_negative.csv` 合并，10,000 条）
- 真值列：`human_review_label`（人工审核，取值 `DATA_SECURITY` / `OTHER`）
- 复算时间：2026-07-22
- 指标实现：`pipeline/step4/metrics.py::binary_metrics`（正类=DATA_SECURITY）

## ⚠️ 口径说明（重要）

这 10,000 条为**分层抽样**（`positive_priority` 5,000 + `negative_priority` 5,000），**不是**线上总体的随机样本。因此下列指标是「在该金标准集合上」的条件表现，**不能直接外推到线上总体分布**。金标准正类占比 48.4%（4,844/10,000），与线上先验（约 31% 正）不同。

金标准标签分布：DATA_SECURITY 4,844 / OTHER 5,156。

## Step 1（关键词法）vs 人工金标准

| 指标 | 值 |
|---|---:|
| Accuracy | 0.6753 |
| Balanced Accuracy | 0.6847 |
| Macro-F1 | 0.6478 |
| DATA_SECURITY Recall | **0.9853** |
| DATA_SECURITY Precision | 0.6005 |
| OTHER Recall / Specificity | **0.3840** |
| OTHER Precision | 0.9654 |

混淆矩阵（行=真值，列=预测）：

| 真值 \ 预测 | OTHER | DATA_SECURITY |
|---|---:|---:|
| OTHER | 1,980 | 3,176 |
| DATA_SECURITY | 71 | 4,773 |

**解读**：关键词法召回率极高（0.985，几乎不漏正例），但精确度低（0.60）——把 3,176 条真负例误召为正（占全部负例 61.6%）。这正是「大量硬负例（S&OTHER）需要 Step 2 教师模型过滤」的来源。

## Step 2（GLM-5.2 标注）vs 人工金标准

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9654 |
| Balanced Accuracy | 0.9659 |
| Macro-F1 | **0.9654** |
| DATA_SECURITY Recall | 0.9804 |
| DATA_SECURITY Precision | 0.9498 |
| OTHER Recall / Specificity | 0.9513 |
| OTHER Precision | 0.9810 |

混淆矩阵（行=真值，列=预测）：

| 真值 \ 预测 | OTHER | DATA_SECURITY |
|---|---:|---:|
| OTHER | 4,905 | 251 |
| DATA_SECURITY | 95 | 4,749 |

### Step 2 分层表现

| 切片 | n | Accuracy |
|---|---:|---:|
| 置信度 ≥ 0.85 | 9,317 | 0.9752 |
| 置信度 < 0.85 | 683 | 0.8316 |
| cohort=positive_priority | 5,000 | 0.9802 |
| cohort=negative_priority | 5,000 | 0.9506 |

Step 1 分层对照：positive_priority 0.5978 / negative_priority 0.7528。

**解读**：
- Step 2 教师模型 Macro-F1 0.965，两类召回均 ≥ 0.95，作为蒸馏教师标签质量可靠。
- 置信度是有效的可靠性信号：<0.85 子集准确率骤降到 0.83（683 条），与「低置信样本应优先人工复核」的策略一致。
- Step 2 相对 Step 1 把负类召回从 0.384 提升到 0.951，是整条流水线过滤硬负例的核心环节。

## 结论

1. Step 1 关键词法：高召回、低精确的粗筛器（负类 Specificity 仅 0.38），必须靠 Step 2 收敛。
2. Step 2 GLM-5.2：Macro-F1 0.965、负类 Recall 0.951，达到作为蒸馏教师的验收线（≥0.90）。
3. 后续小模型 SFT 的验收锚点仍为 Macro-F1 ≥ 0.90、负类 Recall ≥ 0.90（对齐教师）。
