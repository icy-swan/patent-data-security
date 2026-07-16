# Patent Data Security v2

当前版本实现 Step 1 的 S/E 路由，以及 Step 2 的任务绑定、固定法律全文 Prompt、二分类与两组分析标签、火山方舟 Responses 请求和可恢复运行。

```bash
python -m pip install -e '.[dev]'
```

```bash
python -m pipeline.step1 \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step1 \
  --workers 4
```

主要输出：

- `step1_<dataset>.csv`：每件唯一专利一行，含命中位置、上下文、来源、S/E 路由和稳定 E 抽样结果；
- `step1_summary_<dataset>.json`：版本、资源哈希、唯一专利数、重复关联行和路由统计；
- `.step1_<dataset>.partial.sqlite3`：运行中的磁盘去重表，成功后默认删除。

当前词表是 pilot 种子词表，尚不能替代人工开发集和留出集验证。方法与限制见 [docs/patent_identification_methodology.md](docs/patent_identification_methodology.md)。

Step 2 先把 Step 1 已冻结的 `S_all + E_random` 任务池与原始专利正文绑定：

```bash
python -m pipeline.step2 prepare \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --step1-results data/step1/keyword-2.0.0-pilot.2/step1_2021.csv \
  --output-dir data/step2/data-security-binary-v2.1.0
```

确认模型、端点和 API Key 后再发送请求：

```bash
python -m pipeline.step2 run \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step2/data-security-binary-v2.1.0 \
  --model "$ARK_MODEL" \
  --concurrency 10
```

默认并发数为 10。运行终端和 `step2_progress_<dataset>.json` 会持续更新完成数、成功/失败数、本次运行墙钟耗时、累计请求耗时、平均请求耗时、预计剩余秒数和预计完成时间。

从 `v1/.env` 读取方舟配置并在后台启动：

```bash
python -m pipeline.step2 start \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step2/data-security-binary-v2.1.0 \
  --env-file v1/.env \
  --concurrency 10
```

查看状态、安全停止和跟踪终端日志：

```bash
python -m pipeline.step2 status \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step2/data-security-binary-v2.1.0

python -m pipeline.step2 stop \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step2/data-security-binary-v2.1.0

tail -n 50 -f data/step2/data-security-binary-v2.1.0/step2_run_2021.log
```

`stop` 发送 `SIGTERM`：runner 停止领取新任务，等待正在执行的请求完成并落库后退出。日志中的 `Ctrl-C` 只退出 `tail`，不会停止后台分类。

请求方式沿用 V1：用火山方舟官方 `/api/v3/responses` 端点，并通过方舟文档支持的 OpenAI Python SDK 兼容层调用；认证只读取 `ARK_API_KEY`，模型只读取 `ARK_MODEL`。`patent_id` 会进入动态载荷作为不可解释的审计键，但标签结果始终按本地 SQLite 的 `task_id → patent_id` 写回，不依赖模型回传专利号。

`DATA_SECURITY` 结果同时输出 `processing_activities`（收集、存储、使用、加工、传输、提供、公开、其他）和 `industry_sectors`（工业、电信、交通、金融、自然资源、卫生健康、教育、科技、其他）两个受控多标签维度。`OTHER` 的两个维度固定为 `["other"]`；后续子类分析只使用主标签为 `DATA_SECURITY` 的行。Step 2 运行产物位于 `data/step2/`，由 Git 忽略。
