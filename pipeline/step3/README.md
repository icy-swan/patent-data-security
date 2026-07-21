# Step 3：人工标注与开发期模型模拟

新项目从一个或多个已完成的 Step 2 SQLite 数据库冻结 5,000 条样本：

```bash
python -m pipeline.step3 prepare
```

抽样固定为 3,000 条 Step 2 `DATA_SECURITY` 正例和 2,000 条 `Step 1=S → Step 2=OTHER`
难负例；两个抽样组内部按输入年份尽可能等额分配。任一抽样组容量不足时命令失败，
不会从 `E → OTHER` 静默补充容易负例。

抽样完成后的正式目录只有三项：`manifest.json`、`tasks.sqlite3` 和
`need_manual_review.csv`。人工复核 CSV 覆盖全部 5,000 条，展示 Step 2 的标签、置信度、
受控维度、技术/法律范围、逐字证据、`step2_reason` 和不确定性提示；人工填写最后两列
`human_review_label` 与 `human_reason`。人工标签只能是 `DATA_SECURITY` 或 `OTHER`。这轮属于
对 Step 2 结论的人工复核，不是盲标。

开发期使用本机已经登录的 Codex 独立模拟标注，不读取 `OPENAI_API_KEY`：

```bash
python -m pipeline.step3 simulate \
  --model gpt-5.6-sol \
  --reasoning-effort low \
  --batch-size 20
```

程序通过 `codex exec --ephemeral` 使用 ChatGPT/Codex 登录状态，每批结果通过 JSON Schema
校验并即时写入 SQLite。中断后执行同一命令即可续跑。全部成功后只写入
根目录 `simulation.csv`，不会生成训练、验证或测试集。

模拟结果的 `annotation_source=codex_model_simulation`、
`gold_status=provisional_not_human_gold` 且 `eligible_for_final_evaluation=false`。它们只能用于
开发流程、提示词和人工复核辅助，不能替代正式人工结果，也不能进入训练切分。

人工标注完成后，将结果放到根目录 `result.csv`。该文件允许临时带有其他列；执行
`finalize` 时会验证 5,000 条记录与 `tasks.sqlite3` 冻结正文完全对应。除人工最终标签外，
还要求理由、逐字证据和受控维度满足 Step 2 的同一结构化 Schema：

```text
sample_id,dataset_id,application_year,patent_id,title,abstract,claim,ipc,main_ipc,
step1_label,step2_label,step2_confidence,step2_scope_basis,step2_processing_activities,
step2_industry_sectors,step2_technical_scope,step2_legal_scope,step2_evidence,step2_reason,
step2_needs_review,step2_review_reason,human_review_label,human_reason
```

`step1_label`、`step2_label` 和 `human_review_label` 只能是 `DATA_SECURITY` 或 `OTHER`；
`step2_scope_basis`、`step2_processing_activities`、`step2_industry_sectors` 和 `step2_evidence`
使用 JSON。`step2_needs_review` 只表示模型输出是否存在不确定性，不承载类别结论。可先计算
Step 1/2 评估指标：

```bash
python -m pipeline.step3 evaluate
```

`evaluate` 直接以 `human_review_label` 为金标准，不再解释或翻转布尔值；以
`DATA_SECURITY` 为正类，将 Step 1 的 `step1_label` 和 Step 2 的 `step2_label` 分别与金标准比较。
报告写入根目录 `manifest.json` 的 `evaluation`，包含混淆矩阵、Accuracy、
Precision、Recall/Sensitivity、Specificity、NPV、F1、Balanced Accuracy、MCC、Cohen's Kappa
及未加权 Accuracy 的 Wilson 95% 区间。

报告同时提供 `sample_unweighted` 和 `eligible_frame_design_weighted`。后者按年份和抽样组使用
Step 3 纳入概率的倒数加权，只能推广到“Step 2 正例 + `S→OTHER` 难负例”的 9,756 条合格
任务池；由于 `E→OTHER` 未进入人工样本，两种结果都不能表述为全量原始专利的总体准确率。
Step 1 的 `S/E` 本来是路由而非最终分类，其指标仅作诊断。

生成正式 8:1:1 切分：

```bash
python -m pipeline.step3 finalize
```

`finalize` 会在切分完成后自动重新计算并写入上述 Step 1/2 指标。

人工完成前，根目录固定为 `manifest.json`、`tasks.sqlite3` 和 `need_manual_review.csv`。
只有显式执行可选的 `simulate` 时才会暂时产生 `progress.json` 与 `simulation.csv`。
`finalize` 后新增 `result.csv` 和 `dataset/{train,validation,test}.csv`；三个切分文件与
`result.csv` 使用完全相同的结构化人工结果 Schema，不携带抽样概率或请求运行元数据。
