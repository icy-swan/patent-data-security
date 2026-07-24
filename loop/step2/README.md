# Step 2 公开版参考代码

本目录用于论文附录和公开展示。它保留 Step 2 的可复现任务构造、单件识别、严格 JSON 校验、
失败重试和精简结果导出，但不披露任何实际大模型供应商方案。

## 脱敏边界

公开代码不包含：

- 具体模型名称或版本；
- 服务地址、API URL 或 SDK 初始化方式；
- API Key、鉴权头、环境变量名称；
- 生产并发参数、账户配置或计费方式；
- token 消耗、缓存命中、请求耗时、响应 ID；
- 可能包含服务位置或请求标识的原始异常文本；
- 未公开的 Prompt 资源、内部模型回退逻辑或中间结果。

`config.example.json` 中的 Step 1 输入地址为空。`manifest.json` 只保存输入文件名和 SHA-256，
不保存本地绝对路径。

## 公开流程

### 1. 固定 Step 2 任务池

从一个或多个公开版 Step 1 `result.csv` 中读取 `selected_for_step2=true` 的唯一专利，按
固定 seed 的 SHA-256 排序，无放回抽取 `pool_size` 件：

```text
score = SHA256(pool_seed + "|" + patent_id)
```

运行命令：

```bash
python -m loop.step2.step2_public prepare \
  --config loop/step2/config.example.json \
  --step1-result path/to/step1/result.csv \
  --output-dir loop/step2/output
```

准备阶段只生成：

- `requests.jsonl`：每行一件专利，仅含通用任务字段；
- `tasks.sqlite3`：本地断点状态；失败仅记录通用错误码；
- `manifest.json`：输入哈希、抽样规则和字段契约。
- `prompt.txt`：为本次任务冻结的公开 Prompt 副本及其哈希。

准备阶段不会调用任何模型。

### 2. 注入使用者自己的模型适配器

公开仓库只定义以下接口，不提供实现：

```python
class ModelAdapter:
    def classify(self, *, system_prompt: str, patent: dict[str, str]) -> dict:
        ...
```

使用者可在自己的私有模块中创建 factory，并通过 `module:factory` 注入：

```bash
python -m loop.step2.step2_public run \
  --output-dir loop/step2/output \
  --adapter your_private_adapter:create_adapter \
  --concurrency 1
```

主程序每次只把一件专利交给一次 `classify` 调用，不传递前一件专利的上下文；送入适配器
的字段严格限于 `title`、`abstract`、`claim` 和 `ipc`，不会发送任务编号或数据集编号。
供应商地址、认证和请求细节全部留在使用者自己的适配器中，不进入本公开目录。

## Prompt 返回格式

公开 Prompt 只允许三个论文相关字段：

```json
{
  "label": "DATA_SECURITY",
  "reason": "基于专利正文的判定理由",
  "evidence": [
    {
      "field": "claim",
      "quote": "专利原文中的逐字引文"
    }
  ]
}
```

程序严格拒绝额外字段。因此 `confidence`、token、缓存信息、请求耗时、模型名称、服务地址、
响应 ID 等内容即使由适配器返回，也不会进入结果，而是被视为不符合公开 Schema。

## 输出

全部任务成功后才生成 `result.csv`，字段为：

```text
task_id,dataset_id,patent_id,title,abstract,claim,ipc,
step1_route,step2_label,step2_reason,step2_evidence
```

该文件只包含论文分类与审计所需内容。`progress.json` 只记录完成、失败、排队和运行数量。

## 测试

测试使用本地假适配器，不访问网络：

```bash
python -m unittest discover -s loop/step2/tests -v
```
