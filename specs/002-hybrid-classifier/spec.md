# Spec 002：逐条调用火山方舟的数据安全专利三分类

- 状态：Draft
- 版本：2.0.0
- 日期：2026-07-11
- 上游：Step 1 的 S/W/R/E 关键词与上下文检索结果

## 1. 目标

Step 2 对 Step 1 的唯一专利执行大模型三分类：

- S、W、R 层全部进入模型；
- E 层按唯一 `patent_id` 进行稳定 2% 概率抽样，用于估计 Step 1 的漏召；
- 同一申请号关联多个上市公司时只请求一次，避免重复费用；
- 每次只请求一件专利，不使用 Batch JSONL 或批量推理 API；
- 任务可以随时停止、查看结果并从未完成任务继续。

## 2. 模型接入

采用火山方舟 Responses API，基础地址为：

```text
https://ark.cn-beijing.volces.com/api/v3
```

运行环境使用：

- `ARK_API_KEY`：API Key；
- `ARK_MODEL`：具体模型或推理接入点 ID；
- `ARK_BASE_URL`：可选，默认使用北京地域 v3 地址。

`src/patent_data_security/step2_prompt.py` 必须独立提供：

1. `build_classification_prompt`：只负责构造单件专利的分类 Prompt；
2. `VolcengineArkClient.classify`：只执行一次同步模型请求并校验一次响应。

## 3. 输入与抽样

每个年份从以下文件读取 Step 1 路由：

```text
data/step1/keyword_S_<dataset_id>.csv
data/step1/keyword_W_<dataset_id>.csv
data/step1/keyword_R_<dataset_id>.csv
data/step1/keyword_E_<dataset_id>.csv
```

按 S、W、R、E 顺序去重；同一 `patent_id` 若出现在多个层级，保留最高层级。E 层仅在排除
已进入 S/W/R 的专利后抽样。抽样使用固定 seed 与稳定哈希，概率为 0.02，保存：

- `selection_group=E_sample`；
- `selection_probability=0.02`；
- `sample_weight=50`。

S/W/R 的概率和权重均为 1。

## 4. 三分类定义

| 类别 | 定义 |
| --- | --- |
| 1 | 明确数据安全相关：保护对象、风险/约束、技术机制、直接效果与发明中心性形成完整证据链 |
| 2 | 可能数据安全相关但不确定：已有实质性正向证据，但机制、效果或中心性存在关键缺口，需要人工核验 |
| 3 | 其他：不满足类别 1 或 2，不能由专利文本建立实质性数据安全关联 |

分类依据采用“保护对象/处理活动—安全目标或风险—技术机制—直接效果与中心性”证据链。
关键词层级与上下文关联只用于抽样和本地审计，不得进入模型输入。模型只能接收专利名称、摘要、
主权项、IPC 分类号和 IPC 主分类号。类别 2 固定使用 `subtype=potential_data_security`、
`review_flag=true`，并说明待核验的关键缺口；类别 3 固定使用 `subtype=other`。

## 5. 输出契约

模型必须返回单个 JSON 对象，并通过 Pydantic 校验：

- `cat`：1、2、3；
- `confidence`：0 到 1；
- `subtype`：固定枚举；
- `core_invention`：主权项核心技术问题、必要技术手段和直接效果的简要概括；
- `evidence_chain`：分别记录保护对象/活动、安全目标/风险、技术机制、因果中心性和缺失环节；
- `evidence`：1 到 3 条输入原文证据；
- `reason`：核心技术对象与边界判断；
- `review_flag`、`review_reason`。

无效 JSON、非法类别或 subtype/category 冲突都视为失败请求，按配置重试；超过最大次数后记录
为 `failed`，不得伪造标签。

## 6. 可恢复状态与进度

每个数据集使用三个 Step 2 产物：

```text
data/step2/classification_state_<dataset_id>.sqlite3
data/step2/classification_results_<dataset_id>.csv
data/step2/classification_progress_<dataset_id>.json
```

SQLite 是任务状态事实来源。任务状态为 `pending/running/succeeded/failed`；重启时把遗留的
`running` 恢复为 `pending`。每次请求结束后立即提交数据库、更新 CSV 和进度 JSON。

进度至少包含：

- 总任务数、完成数、成功数、失败数、待处理数和百分比；
- 请求使用的模型与实际返回模型；
- 平均请求耗时、平均完成任务耗时和预计剩余秒数；
- 最近更新时间。

## 7. 后台运行与停止

`scripts/step2_llm_classification.py` 提供：

- `prepare`：只创建任务，不调用模型；
- `run`：前台逐条请求；
- `start`：启动独立后台进程；
- `status`：读取 PID 和各数据集进度；
- `stop`：发送停止信号，当前请求结束后安全退出。

十年原始 CSV 默认从 `data/raw/*.csv` 自动发现并按文件名中的年份形成 `dataset_id`，按年份顺序
串行处理，避免十个后台进程同时消耗配额。

## 8. 验收标准

- [x] S/W/R 唯一专利全部进入任务库；
- [x] E 唯一专利按稳定 2% 概率抽样并保存权重 50；
- [x] 相同行号出现在不同年份时，`task_id` 不冲突；
- [x] Prompt 构建与 API 调用相互独立；
- [x] 单元测试使用假客户端，不产生真实模型请求；
- [x] 中断后不会重复请求已经成功或最终失败的任务；
- [x] 进度文件包含模型、平均耗时和 ETA；
- [x] 不生成 Batch JSONL 文件；
- [ ] 配置真实方舟模型后完成少量在线 smoke test。
