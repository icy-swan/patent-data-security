# Step 3 公开版参考代码

本目录复现论文 Step 3 从 50,000 条 Step 2 结果中抽取 10,000 条人工复核样本的逻辑。
这里只包含确定性抽样，不包含自动模型复核、模型调用或模型效果评测。

## 抽样组定义

每条 Step 2 结果根据前两步标签进入一个抽样组：

| 抽样组 | 定义 |
|---|---|
| `positive` | `step2_label=DATA_SECURITY` |
| `hard_negative` | `step1_route=S` 且 `step2_label=OTHER` |
| `easy_negative` | `step1_route=E` 且 `step2_label=OTHER` |

## 两个互斥队列

公开配置按顺序生成两个互不重叠的 5,000 条队列：

| 队列 | positive | hard_negative | easy_negative | 合计 |
|---|---:|---:|---:|---:|
| `positive_priority` | 3,000 | 2,000 | 0 | 5,000 |
| `negative_priority` | 2,000 | 1,000 | 2,000 | 5,000 |
| 合计 | 5,000 | 3,000 | 2,000 | 10,000 |

第二个队列只从第一个队列未选中的专利中抽取，因此两者不会重复。

## 年份均衡

每个“队列 × 抽样组”内部先按 `application_year` 分层，再将目标数尽可能均匀分给各年份。
某年份容量不足时，多出的名额会确定性地重新分配给仍有容量的年份。年份内按以下哈希排序后
取前若干条：

```text
score = SHA256(seed + "|" + dataset_id + "|" + patent_id)
```

申请年份优先使用 Step 2 的 `application_year`，其次从 `application_date` 或 `dataset_id`
提取四位年份；仍无法识别时进入单独的 `UNKNOWN` 层。

## 输入

`config.example.json` 中的 `step2_result` 有意留空。输入必须是完整的 Step 2 `result.csv`，
默认严格要求 50,000 条且 `patent_id` 唯一。论文复现时不应通过修改
`expected_population_size` 绕过完整样本框检查。

运行：

```bash
python -m loop.step3.step3_public \
  --config loop/step3/config.example.json \
  --step2-result path/to/step2/result.csv \
  --output-dir loop/step3/output
```

## 输出

- `need_manual_review_positive.csv`：正例优先队列；
- `need_manual_review_negative.csv`：负例优先补充队列；
- `review_sample.csv`：两个队列合并后的 10,000 条样本；
- `manifest.json`：输入哈希、分层容量、配额、纳入概率和输出哈希。

CSV 保留 Step 1/2 标签、Step 2 理由和逐字证据，并将 `human_review_label`、`human_reason`
留空供人工复核。若 Step 2 提供了累计纳入概率，代码还会计算从前序样本框到 Step 3 的累计
概率和逆概率权重。

`manifest.json` 只记录输入文件名和 SHA-256，不保存本地绝对路径。整个准备过程不会发起
任何模型请求。
