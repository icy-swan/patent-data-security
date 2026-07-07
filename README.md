# patent-data-security

用 Python 调用 LLM 处理 CSV 专利数据，识别专利是否属于数据安全领域。

## 项目目标

- 读取专利 CSV 数据。
- 使用 LLM 对专利标题、摘要、权利要求等字段进行领域识别。
- 输出结构化判断结果，便于后续复核、统计和分析。

## 目录结构

```text
.
├── config/                 # 本地配置模板
├── data/
│   ├── raw/                # 原始 CSV 数据，不提交真实数据
│   ├── interim/            # 清洗或抽样后的中间数据
│   └── processed/          # 最终处理结果
├── docs/                   # 项目说明和设计记录
├── prompts/                # LLM 提示词模板
├── src/
│   └── patent_data_security/
│       └── __init__.py     # Python 包入口
├── tests/                  # 测试
├── .env.example            # 环境变量示例
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## 本地初始化

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

## 数据约定

真实 CSV 数据默认放在 `data/raw/` 下，不提交到 Git。

建议后续统一输入字段，例如：

- `publication_number`
- `title`
- `abstract`
- `claims`
- `applicant`
- `publication_date`

## 输出约定

建议后续输出到 `data/processed/`，并保留以下识别字段：

- `is_data_security`
- `confidence`
- `reason`
- `matched_topics`
- `model`
- `processed_at`

