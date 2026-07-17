# Step 3：人工标注与开发期模型模拟

先从一个或多个已完成的 Step 2 SQLite 数据库冻结 4,000 条样本：

```bash
python -m pipeline.step3 prepare \
  --step2-dir data/step2/data-security-binary-v2.1.0 \
  --output-dir data/step3/positive-priority-v2.2.0
```

抽样固定为 3,000 条 Step 2 `DATA_SECURITY` 正例和 1,000 条 `Step 1=S → Step 2=OTHER`
难负例；两个抽样组内部按输入年份尽可能等额分配。任一抽样组容量不足时命令失败，
不会从 `E → OTHER` 静默补充容易负例。

`sample/annotation_input.csv` 不含 Step 2 标签、置信度、路由或模型解释，可直接作为
独立人工标注模板。`sample/audit.csv` 保存抽样层、两阶段纳入概率和评估权重，两者不得
在首轮标注前合并展示。

开发期使用本机已经登录的 Codex 独立模拟标注，不读取 `OPENAI_API_KEY`：

```bash
python -m pipeline.step3 simulate \
  --output-dir data/step3/positive-priority-v2.2.0 \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --batch-size 20
```

程序通过 `codex exec --ephemeral` 使用 ChatGPT/Codex 登录状态，每批结果通过 JSON Schema
校验并即时写入 SQLite。中断后执行同一命令即可续跑。全部成功后自动生成严格
3,200/400/400 的 train/validation/test 文件。

模拟结果的 `annotation_source=codex_model_simulation`、
`gold_status=provisional_not_human_gold` 且 `eligible_for_final_evaluation=false`。它们只能用于
开发流程、提示词和训练试验，不能替代正式人工金标准或用于最终性能报告。

产物按用途分为三个目录：`sample/` 保存冻结样本、盲标输入与抽样清单，`state/` 保存可恢复
任务库与进度，`dataset/` 保存唯一一份 provisional 全量数据、切分报告以及
`splits/{train,validation,test}.csv`。文件所在目录已经表达 Step 3 语义，因此文件名不再重复
`step3_` 前缀。
