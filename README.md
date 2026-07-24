# Patent Data Security v2

当前版本实现 Step 1 的 S/E 路由，以及 Step 2 的任务绑定、固定法律全文 Prompt、二分类与两组分析标签、火山方舟 Responses 请求和可恢复运行。

```bash
python -m pip install -e '.[dev]'
```

```bash
python -m pipeline.step1 \
  --input data/raw/上市企业专利明细/上市公司专利明细_2021年申请.csv \
  --output-dir data/step1 \
  --workers 4
```

主要输出：

- `data/step1/<年份>/result.csv`：每件唯一专利一行，含命中位置、上下文、来源、S/E 路由和稳定 E 抽样结果；
- `data/step1/<年份>/manifest.json`：版本、资源哈希、唯一专利数、重复关联行和路由统计。

运行中使用的 SQLite 去重表在成功后自动删除，不属于正式产物。

当前词表是 pilot 种子词表，尚不能替代人工开发集和留出集验证。方法与限制见 [docs/patent_identification_methodology.md](docs/patent_identification_methodology.md)。

Step 1 共输出 224,907 条 `S_all + E_random` 候选记录。Step 2 先跨年份按
`patent_id` 去重为 224,906 件唯一专利，再用固定 seed 的 SHA-256 顺序做等概率无放回抽样，
固定保留 50,000 件：

```bash
python -m pipeline.step2 prepare-pool \
  --raw-dir data/raw/上市企业专利明细 \
  --step1-dir data/step1 \
  --output-dir data/step2 \
  --pool-size 50000 \
  --pool-seed step2-global-pool-v1 \
  --rebuild
```

这条命令只准备任务，不调用模型。正式产物直接写入 `data/step2/`：

- `requests.jsonl`：50,000 条待识别动态载荷的离线审计副本，一行一件专利，不作为批量请求体；
- `tasks.sqlite3`：本地任务绑定、状态、两阶段抽样概率和恢复信息；
- `manifest.json`：来源哈希、全局去重、抽样参数和分层计数。

二次纳入概率为 `50000 / 224906`。最终 `selection_probability` 等于 Step 1 纳入概率乘以
该二次概率，`sample_weight` 是其倒数。实际样本含 `S_all` 21,711 件、`E_random`
28,289 件；未强制年份或路由配额。程序仍保留单年份 `prepare --input ...` 入口。

收到明确识别命令后，才可以发送请求：

```bash
python -m pipeline.step2 run \
  --output-dir data/step2 \
  --model glm-5-2-260617 \
  --concurrency 10
```

runner 从 `tasks.sqlite3` 每次领取一件专利，为它单独构造“固定 Prompt 前缀 + 当前专利动态
后缀”，并单独调用一次 Responses API。请求之间不传递历史消息，也不把 `requests.jsonl`
整包发送给模型。默认并发数为 10，表示最多同时执行 10 个彼此独立的单件请求。运行终端和
`progress.json` 会持续更新完成数、成功/失败数、本次运行墙钟耗时、累计请求耗时、
平均请求耗时、预计剩余秒数和预计完成时间。

从仓库根目录 `.env` 读取 Agent Plan Key 并在后台启动：

```bash
python -m pipeline.step2 start \
  --output-dir data/step2 \
  --env-file .env \
  --model glm-5-2-260617 \
  --concurrency 10
```

查看状态、安全停止和跟踪终端日志：

```bash
python -m pipeline.step2 status \
  --output-dir data/step2

python -m pipeline.step2 stop \
  --output-dir data/step2

tail -n 50 -f data/step2/runner.log
```

`stop` 发送 `SIGTERM`：runner 停止领取新任务，等待正在执行的请求完成并落库后退出。日志中的 `Ctrl-C` 只退出 `tail`，不会停止后台分类。

请求方式沿用 V1：用火山方舟官方 `/api/v3/responses` 端点，并通过方舟文档支持的 OpenAI Python SDK 兼容层调用；认证只读取 `ARK_API_KEY`，模型由 `--model` 或 `ARK_MODEL` 指定。`patent_id` 会进入动态载荷作为不可解释的审计键，但标签结果始终按本地 SQLite 的 `task_id → patent_id` 写回，不依赖模型回传专利号。

`DATA_SECURITY` 结果同时输出 `processing_activities`（收集、存储、使用、加工、传输、提供、公开、其他）和 `industry_sectors`（工业、电信、交通、金融、自然资源、卫生健康、教育、科技、其他）两个受控多标签维度。`OTHER` 的两个维度固定为 `["other"]`；后续子类分析只使用主标签为 `DATA_SECURITY` 的行。

本轮 Step 2 使用跨年份固定任务池，产物直接保存在 `data/step2/`，完成后目录内保留
`requests.jsonl`、`result.csv`、`manifest.json`、`tasks.sqlite3` 和 `progress.json`。
后台运行期间产生的
`runner.pid`、`runner.log`、锁文件及 SQLite sidecar 会在全部任务成功后删除。

Step 3 采用两个互不重叠的 5,000 条队列：第一批按 Step 2 预测正/负 3:2 抽取，第二批按
2:3 抽取，其中固定包含 1,000 条 `Step1=DATA_SECURITY → Step2=OTHER` 难负例和 2,000 条
`Step1=OTHER → Step2=OTHER` 容易负例。两批合并后按 Step 2 预测标签为 1:1，并覆盖完整
50,000 条 Step 2 任务池的三个抽样组；真实 Gold 比例以人工复核结果为准。详细命令见
[pipeline/step3/README.md](pipeline/step3/README.md)。

人工复核输入为 `data/step3/need_manual_review_positive.csv` 和
`data/step3/need_manual_review_negative.csv`，其中包含 `step1_label`、
`step2_label`、逐字证据和大模型决策理由，人工填写 `human_review_label` 与
`human_reason`。三个标签字段都只使用 `DATA_SECURITY/OTHER`，不再通过布尔值推导类别。

两批结果分别保存为 `result_positive.csv` 和 `result_negative.csv`，通过严格去重与冻结字段
校验后合并为 `result.csv`：

```bash
python -m pipeline.step3 merge
python -m pipeline.step3 finalize
```

只有完成 10,000 条人工核验并通过合并校验的 `result.csv` 才会进入
`data/step3/dataset/` 下的训练、验证和测试切分；Codex 模拟结果不生成训练数据。`evaluate`
会把 Step 1/2 的混淆矩阵、Accuracy、
Precision、Recall、Specificity、F1 等样本指标和设计加权指标写入 `manifest.json`；`finalize`
也会自动刷新这些指标。

Step 4 从冻结切分生成 RoBERTa 分类数据和 MaaS `messages` JSONL：

```bash
python -m pipeline.step4 prepare
```

只在本地训练 RoBERTa；SFT JSONL 由用户上传 MaaS，仓库不实现 SFT 训练：

```bash
python -m pip install -e '.[step4]'

python -m pipeline.step4 train-roberta \
  --output-dir data/step4 \
  --model hfl/chinese-roberta-wwm-ext \
  --text-fields abstract
```

产物说明和显存参数见 [pipeline/step4/README.md](pipeline/step4/README.md)。
