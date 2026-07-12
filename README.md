# patent-data-security

面向中国上市公司专利的可审计数据安全识别管线。系统先对摘要和主权项做带上下文关联的
S/W/R/E 关键词路由，再逐件调用火山方舟大模型输出三分类结果：

1. 明确数据安全相关；
2. 可能数据安全相关但不确定；
3. 其他。

S/W/R/E 是召回路由，不是最终标签。S/W/R 唯一专利全量进入模型；E 唯一专利按稳定 2%
概率抽样进入模型，用于估计关键词体系的漏召。

## 方法文档

- `docs/patent_identification_methodology.md`：从关键词召回、GLM-5.2 初分类、人工 Gold、Alpha
  Silver 构建，到 RoBERTa/SFT 模型竞赛和 E 层认证的完整研究方法；
- `specs/001-docs-ipc-taxonomy/spec.md`：DOCS/IPC 分层定义与来源要求；
- `specs/001-docs-ipc-taxonomy/research.md`：法规、标准、论文和附件方法研究；
- `specs/002-hybrid-classifier/spec.md`：逐条火山方舟分类、抽样、恢复和进度契约；
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

当前只保留两个步骤：

| 步骤 | 脚本 | 默认产物 | 是否调用外部 LLM |
| --- | --- | --- | --- |
| 1 | `scripts/step1_keyword_extraction.py` | `data/step1/` 下的 S/W/R/E 四个关键词文件及汇总 | 否 |
| 2 | `scripts/step2_llm_classification.py` | `data/step2/` 下的 SQLite 状态、逐条结果和进度 | `run/start` 会调用 |

项目只保留上述分步脚本作为执行入口，避免旧聚合 CLI 与新目录契约并存。

Step 2 的接入方式遵循[火山方舟官方 Responses API 示例](https://www.volcengine.com/docs/82379/1795150)，
使用 OpenAI Python SDK 兼容接口和北京地域 `/api/v3` 基础地址。

## Step 1：关键词与上下文关联检索

Step 1 仅扫描 `摘要文本`和`主权项内容`，不读取 IPC 作为关键词证据，也不导入或调用 LLM。
上下文检索优先使用命中词所在的完整句子；字段没有可靠句界时，退回命中词左右各 48 字窗口。
每个关键词命中会保存上下文词表 ID、类型、命中词、位置、距离、片段、来源和检索范围。

```bash
.venv/bin/python scripts/step1_keyword_extraction.py --workers 4
```

脚本默认扫描 `data/raw/*.csv`。以后加入十个年度文件后同一条命令会逐年处理；已完整生成的年份
默认跳过。也可以重复传入 `--input` 只处理指定文件。

默认产物：

- `data/step1/keyword_S_2021.csv`
- `data/step1/keyword_W_2021.csv`
- `data/step1/keyword_R_2021.csv`
- `data/step1/keyword_E_2021.csv`
- `data/step1/keyword_summary_2021.json`

S/W/R/E 是关键词召回强度；E 表示关键词体系未路由，不是最终“不相关”标签。

## Step 2：逐条火山方舟分类

先配置：

```bash
cp .env.example .env
# 在 .env 中设置 ARK_API_KEY 和 ARK_MODEL
```

只准备任务，不发请求：

```bash
.venv/bin/python scripts/step2_llm_classification.py prepare
```

前台逐条请求：

```bash
.venv/bin/python scripts/step2_llm_classification.py run
```

后台启动、查看、停止：

```bash
.venv/bin/python scripts/step2_llm_classification.py start
.venv/bin/python scripts/step2_llm_classification.py status
.venv/bin/python scripts/step2_llm_classification.py stop
```

每次请求后立即写入 SQLite、CSV 和进度 JSON。再次运行 `run/start` 会从未完成任务继续，不会
重复请求成功任务。该流程不生成 Batch JSONL。

## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
```
