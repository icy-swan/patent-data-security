# Step 3：人工标注与开发期模型模拟

新项目从一个或多个已完成的 Step 2 SQLite 数据库冻结 5,000 条样本：

```bash
python -m pipeline.step3 prepare
```

抽样固定为 3,000 条 Step 2 `DATA_SECURITY` 正例和 2,000 条 `Step 1=S → Step 2=OTHER`
难负例；两个抽样组内部按输入年份尽可能等额分配。任一抽样组容量不足时命令失败，
不会从 `E → OTHER` 静默补充容易负例。

已有 4,000 条 v2.2.0 样本时使用增量扩展，不重建旧任务：

```bash
python -m pipeline.step3 expand
```

该命令保留原 3,000 条正向候选和 1,000 条难负向候选，再增加 1,000 条难负例；新增的盲标
输入写入 `annotation_increment.csv`。抽样配额、输入数据库和分层汇总写入根目录
`manifest.json`；5,000 条冻结正文和模拟任务状态
保存在 `tasks.sqlite3`，不再额外输出会干扰阅读的抽样过程 CSV。
扩容前的 3,200/400/400 旧切分归档到 `archive/human-split-v2.2.0/`，不得用于新版 Step 4。

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
human_evaluation,confidence,scope_basis,processing_activities,industry_sectors,technical_scope,legal_scope,
evidence,reason,review_flag,review_reason
```

`human_evaluation` 只能是 `true` 或 `false`，分别表示数据安全正类和负类；
`scope_basis`、`processing_activities`、`industry_sectors` 和 `evidence` 使用 JSON。理由和证据
必须与人工最终标签一致。可先计算 Step 1/2 评估指标：

```bash
python -m pipeline.step3 evaluate
```

`evaluate` 以 `DATA_SECURITY` 为正类，Step 1 将 `S` 视为正预测、`E` 视为负预测，Step 2 直接
使用其二分类标签。报告写入根目录 `manifest.json` 的 `evaluation`，包含混淆矩阵、Accuracy、
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

人工完成前，根目录固定为 `simulation.csv`、`manifest.json`、`tasks.sqlite3`、`progress.json`，
并等待 `result.csv`。`finalize` 后新增 `dataset/{train,validation,test}.csv`，切分报告合并写入
根目录 `manifest.json`。三个切分文件与 `result.csv` 使用完全相同的结构化人工结果 Schema，
不携带前序模型、抽样概率或请求运行元数据。
