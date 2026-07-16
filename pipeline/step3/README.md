# Step 3：人工标注与开发期模型模拟

先从一个或多个已完成的 Step 2 SQLite 数据库冻结 4,000 条样本：

```bash
python -m pipeline.step3 prepare \
  --step2-dir data/step2/data-security-binary-v2.1.0 \
  --output-dir data/step3/step3-balanced-v2.2.0
```

抽样固定为 Step 2 的 `DATA_SECURITY`/`OTHER` 各 2,000 条；每个标签内部按输入年份
尽可能等额分配。任一标签总量不足 2,000 时命令失败，不会静默改变比例。

`step3_annotation_input_blinded.csv` 不含 Step 2 标签、置信度、路由或模型解释，可直接作为
独立人工标注模板。`step3_sample_audit.csv` 保存抽样层、两阶段纳入概率和评估权重，两者不得
在首轮标注前合并展示。

开发期可用 GPT-5.6-sol 独立模拟标注：

```bash
export OPENAI_API_KEY='...'
python -m pipeline.step3 simulate \
  --output-dir data/step3/step3-balanced-v2.2.0 \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --concurrency 5
```

程序按专利逐条请求、禁用 API 端存储、使用 Pydantic Structured Outputs，并将每条结果即时
写入 SQLite。中断后执行同一命令即可续跑。全部成功后自动生成严格 3,200/400/400 的
train/validation/test 文件。

模拟结果的 `annotation_source=openai_model_simulation`、
`gold_status=provisional_not_human_gold` 且 `eligible_for_final_evaluation=false`。它们只能用于
开发流程、提示词和训练试验，不能替代正式人工金标准或用于最终性能报告。
