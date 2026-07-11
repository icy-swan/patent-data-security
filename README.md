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

## 分步执行结构

每一步都有独立脚本和独立产物目录，避免把关键词、IPC、候选构建和外部请求混在一个入口中：

| 步骤 | 脚本 | 默认产物 | 是否调用外部 LLM |
| --- | --- | --- | --- |
| 1 | `scripts/step1_keyword_extraction.py` | `data/step1/` 下的 S/W/R/E 四个关键词文件及汇总 | 否 |
| 2 | `scripts/step2_candidate_routing.py` | `data/step2/` 下的 DOCS/IPC 路由、候选和汇总 | 否 |
| 3 | `scripts/step3_prepare_llm_batches.py` | `data/step3/` 下的 Batch 请求文件和清单 | 否 |
| 4 | `scripts/step4_submit_llm_batch.py` | `data/step4/` 下的提交回执 | **是** |
| 5 | `scripts/step5_merge_llm_results.py` | `data/step5/` 下的三分类结果 | 否 |
| 6 | `scripts/step6_audit_results.py` | `data/step6/` 下的审计报告 | 否 |

项目只保留上述分步脚本作为执行入口，避免旧聚合 CLI 与新目录契约并存。

## Step 1：关键词与上下文关联检索

Step 1 仅扫描 `摘要文本`和`主权项内容`，不读取 IPC 作为关键词证据，也不导入或调用 LLM。
上下文检索优先使用命中词所在的完整句子；字段没有可靠句界时，退回命中词左右各 48 字窗口。
每个关键词命中会保存上下文词表 ID、类型、命中词、位置、距离、片段、来源和检索范围。

```bash
.venv/bin/python scripts/step1_keyword_extraction.py --workers 4
```

默认产物：

- `data/step1/keyword_S_2021.csv`
- `data/step1/keyword_W_2021.csv`
- `data/step1/keyword_R_2021.csv`
- `data/step1/keyword_E_2021.csv`
- `data/step1/keyword_summary_2021.json`

S/W/R/E 是关键词召回强度；E 表示关键词体系未路由，不是最终“不相关”标签。

## Step 2：DOCS/IPC 候选路由

```bash
.venv/bin/python scripts/step2_candidate_routing.py --workers 4
```

中断后增加 `--resume`；确认重跑时使用 `--overwrite`。同一申请号可能关联多个上市公司：
全量路由表保留每个公司行，LLM 候选按申请号去重，并通过 `classification_key` 回连。

重构前的 2021 年路由基线如下；上下文逻辑升级后应使用 Step 2 重新生成，不应与本次 Step 1
关键词分层结果混用：

- CSV 逻辑记录：985,759；
- S/W/R 路由：35,190 行；
- E 分层抽样：2,008 行；
- 模型候选关联行：37,198；
- 去重后的唯一模型候选：18,094。

产物位于 `data/step2/`：

- `patent_routes_2021.csv`
- `patent_llm_candidates_2021.jsonl`
- `route_summary_2021.json`

## Step 3：仅生成请求文件

```bash
.venv/bin/python scripts/step3_prepare_llm_batches.py --model "$LLM_MODEL"
```

该步骤只生成本地 JSONL，不调用 API。

## Step 4：提交 LLM Batch（唯一外部请求步骤）

先在 `.env` 配置 `OPENAI_API_KEY`、可选 `OPENAI_BASE_URL` 和 `LLM_MODEL`。只有显式执行以下
脚本才会产生外部调用和费用：

```bash
.venv/bin/python scripts/step4_submit_llm_batch.py \
  --file data/step3/batch_0001.jsonl
```

## Step 5：合并已下载结果

```bash
.venv/bin/python scripts/step5_merge_llm_results.py \
  --outputs data/step4/results/*.jsonl \
  --model "$LLM_MODEL"
```

失败响应只记录为 `failed`，不会伪造标签。未经人工 Gold Set 校准的结果应称为机器标签，
不能直接作为论文最终标签。

## Step 6：一致性审计

```bash
.venv/bin/python scripts/step6_audit_results.py
```

## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
```
