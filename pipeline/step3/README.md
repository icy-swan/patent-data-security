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
校验并即时写入 SQLite。中断后执行同一命令即可续跑。全部成功后只写入
`dataset/simulation.csv`，不会生成训练、验证或测试集。

模拟结果的 `annotation_source=codex_model_simulation`、
`gold_status=provisional_not_human_gold` 且 `eligible_for_final_evaluation=false`。它们只能用于
开发流程、提示词和人工复核辅助，不能替代正式人工结果，也不能进入训练切分。

人工标注完成后，将结果放到 `dataset/results.csv`。该文件允许临时带有其他列；执行
`finalize` 时会验证 4,000 条记录与冻结盲标输入完全对应，并删除所有非训练字段：

```text
sample_id,dataset_id,application_year,patent_id,title,abstract,claim,ipc,main_ipc,
human_evaluation,scope_basis,industry_sectors
```

`human_evaluation` 只能是 `true` 或 `false`，分别表示数据安全正类和负类；
`scope_basis`、`industry_sectors` 使用 JSON 数组。生成正式 8:1:1 切分：

```bash
python -m pipeline.step3 finalize \
  --output-dir data/step3/positive-priority-v2.2.0
```

产物按用途分为三个目录：`sample/` 保存冻结样本、盲标输入与抽样清单，`state/` 保存可恢复
任务库与进度，`dataset/` 保存模拟审计文件、人工 `results.csv`、切分报告以及由人工结果生成的
`splits/{train,validation,test}.csv`。三个切分文件与 `results.csv` 使用完全相同的 12 列精简
Schema，不携带前序模型、抽样或复核过程字段。
