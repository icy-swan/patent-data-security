# Patent Data Security v2

当前版本实现 Step 1 的 S/E 路由，以及 Step 2 的任务绑定、固定法律全文 Prompt、二分类 Schema、OpenAI 兼容请求和可恢复运行。

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
  --output-dir data/step2/data-security-binary-v2.0.0
```

确认模型、端点和 API Key 后再发送请求：

```bash
python -m pipeline.step2 run \
  --input v1/data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/step2/data-security-binary-v2.0.0 \
  --model "$ARK_MODEL" \
  --concurrency 4
```

`patent_id` 会进入动态载荷作为不可解释的审计键，但标签结果始终按本地 SQLite 的 `task_id → patent_id` 写回，不依赖模型回传专利号。Step 2 运行产物位于 `data/step2/`，由 Git 忽略。
