# Patent Data Security v2

当前版本实现 Step 1：使用版本化关键词与局部上下文规则，把唯一专利路由为 `S`（有效关键词命中）或 `E`（else）。IPC 只生成审计标记，不直接改变路由。

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

