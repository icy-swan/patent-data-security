# Step 3：双队列人工复核

Step 3 采用两个互不重叠的 5,000 条人工复核队列。文件名中的 `positive`、`negative`
表示抽样倾向，不表示人工金标准。

| 队列 | Step 2 预测正例 | Step 2 预测负例 | 负例组成 |
| --- | ---: | ---: | --- |
| `positive_priority` | 3,000 | 2,000 | `Step1=DATA_SECURITY → Step2=OTHER` 难负例 |
| `negative_priority` | 2,000 | 3,000 | 1,000 条难负例 + 2,000 条 `Step1=OTHER → Step2=OTHER` 容易负例 |
| 合并 | 5,000 | 5,000 | 3,000 条难负例 + 2,000 条容易负例 |

上述比例只能约束抽样时的 Step 2 预测标签。第二队列经过人工复核后的真实正负比例事先未知，
以 `human_review_label` 为准。代码只接受冻结的 50,000 条 Step 2 任务池；当前任务池包含
14,631 条预测正例、7,324 条难负例和 28,045 条容易负例。三个抽样组均保留非零纳入概率，
以便人工复核后使用 manifest 中的设计权重估计完整 Step 2 任务池指标。

新项目先冻结正向优先队列：

```bash
python -m pipeline.step3 prepare
```

在第一队列冻结后，追加与其 `sample_id`、`patent_id` 均不重叠的负向优先队列：

```bash
python -m pipeline.step3 prepare-negative
```

人工复核输入为：

- `need_manual_review_positive.csv`
- `need_manual_review_negative.csv`

两份文件都展示 Step 1/2 标签、Step 2 置信度、受控维度、逐字证据、完整决策理由与
不确定性提示。人工只填写 `human_review_label` 和 `human_reason`；标签只能是
`DATA_SECURITY` 或 `OTHER`。`sample_cohort` 用于防止两批数据混淆。

第一批已复核结果保存为 `result_positive.csv`。第二批复核完成后保存为
`result_negative.csv`，再执行：

```bash
python -m pipeline.step3 merge
```

`merge` 不做简单拼接。它会验证每批恰好 5,000 条、队列标识正确、人工标签和理由完整、
冻结正文未变、样本 ID 与专利 ID 在批内和批间都唯一；全部通过后才原子生成 10,000 条
`result.csv`，并把输入/输出哈希、标签统计和合并时间写入 `manifest.json`。

随后生成正式 8:1:1 切分并重新评估 Step 1/2：

```bash
python -m pipeline.step3 finalize
```

输出为 `dataset/train.csv`、`dataset/validation.csv`、`dataset/test.csv`，记录数固定为
8,000 / 1,000 / 1,000。切分按“年份 × 人工最终标签”近似分层，并防止完全相同的专利文本
跨集合。比较不同训练样本配比时，应共用同一份冻结验证集和测试集；不能直接比较由不同
正负分布测试集得到的 Accuracy。

`evaluate` 可单独重算以 `human_review_label` 为金标准的 Step 1/2 指标：

```bash
python -m pipeline.step3 evaluate
```

设计加权指标覆盖完整 Step 2 任务池；它仍不能自动推广到进入 Step 2 之前已被排除的全部
原始专利。
