# patent-data-security

面向中国上市公司专利的可审计数据安全识别管线。系统先对摘要、主权项和 IPC 做
S/W/R/E 候选路由，再由 LLM 按统一 JSON Schema 输出三分类结果：

1. 数据安全相关；
2. 安全相关但非数据安全；
3. 不相关。

S/W/R/E 是召回路由，不是最终标签。S/W/R 全量进入模型；E 通过确定性分层概率抽样进入
模型和人工复核，用于估计关键词体系的漏召。

## 方法文档

- `specs/001-docs-ipc-taxonomy/spec.md`：DOCS/IPC 分层定义与来源要求；
- `specs/001-docs-ipc-taxonomy/research.md`：法规、标准、论文和附件方法研究；
- `specs/002-hybrid-classifier/spec.md`：混合分类、抽样、Batch、Gold Set 与验收口径；
- `config/taxonomy/`：可版本化词项、IPC 规则和逐项来源注册表。

## 初始化

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

真实数据与派生结果均已被 Git ignore。

## 2021 年路由

```bash
python -m patent_data_security route \
  --input data/raw/上市公司专利明细_2021年申请.csv \
  --output-dir data/interim/2021 \
  --workers 4 \
  --checkpoint-every 50000 \
  --progress-every 100000
```

中断后增加 `--resume`；确认重跑时使用 `--overwrite`。同一申请号可能关联多个上市公司：
全量路由表保留每个公司行，LLM 候选按申请号去重，并通过 `classification_key` 回连。

当前 2021 年本地结果：

- CSV 逻辑记录：985,759；
- S/W/R 路由：35,190 行；
- E 分层抽样：2,008 行；
- 模型候选关联行：37,198；
- 去重后的唯一模型候选：18,094。

产物位于 `data/interim/2021/`：

- `patent_routes_2021.csv`；
- `patent_llm_candidates_2021.jsonl`；
- `route_summary_2021.json`；
- `route_audit_2021.json`。

重新审计：

```bash
python -m patent_data_security audit \
  --routes data/interim/2021/patent_routes_2021.csv \
  --candidates data/interim/2021/patent_llm_candidates_2021.jsonl \
  --output data/interim/2021/route_audit_2021.json
```

## LLM Batch

先在 `.env` 配置 `OPENAI_API_KEY`、可选 `OPENAI_BASE_URL` 和 `LLM_MODEL`。生成请求文件不会
调用 API：

```bash
python -m patent_data_security prepare-batches \
  --candidates data/interim/2021/patent_llm_candidates_2021.jsonl \
  --output-dir data/interim/2021/batches \
  --model "$LLM_MODEL"
```

提交单个拆分文件会产生外部调用和费用：

```bash
python -m patent_data_security submit-batch \
  --file data/interim/2021/batches/batch_0001.jsonl
```

下载 Batch 输出后，按 `custom_id` 合并并进行 Pydantic 校验：

```bash
python -m patent_data_security merge-batches \
  --outputs data/interim/2021/batch-results/*.jsonl \
  --destination data/processed/patent_classifications_2021.csv \
  --model "$LLM_MODEL"
```

失败响应只记录为 `failed`，不会伪造标签。未经人工 Gold Set 校准的结果应称为机器标签，
不能直接作为论文最终标签。

## 验证

```bash
python -m pytest -q
ruff check src tests
```
