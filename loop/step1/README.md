# Step 1 公开版参考代码

本目录用于论文附录和公开展示，复现第一阶段的规则路由与概率抽样，同时排除不属于公开范围
的项目资产。该步骤完全在本地运行，不调用大模型，也不发起网络请求。

## 公开方法

1. 对文本执行 Unicode NFKC、拉丁字符大小写折叠、空白和连接符归一化；
2. 按最长词组优先，在 `claim`、`abstract` 和 `title` 中匹配；
3. 同时支持独立命中规则和局部语境共现规则；
4. 支持明确排除短语及仅用于诊断、不改变路由的模式；
5. IPC 仅用于审计，不单独改变 S/E 路由；
6. 每件专利只保留一条记录，重复记录中优先保留 S，其次选择文本更完整者；
7. 保留全部 S，并用固定 seed 的 SHA-256 对 E 做可复现概率抽样；
8. 输出结果 CSV 以及包含计数、抽样参数和文件哈希的 manifest。

## 脱敏边界

公开目录不包含：

- 生产关键词分类体系及其变体；
- 专家词表和未发表的验证样本；
- 原始专利数据及其存储地址；
- 私有来源清单、凭证和项目中间结果。

因此，随代码发布的 `rules.template.json` 中各规则列表均为空。这样可以避免把占位词表误解为
论文实际使用并完成验证的生产词表。

## 输入

`config.example.json` 中原始数据地址有意留空：

```json
{
  "input_csv": ""
}
```

请复制该文件为本地未跟踪配置，再填写自己的 CSV 路径和字段映射；也可以在命令行传入输入
文件。规范字段为：

```text
patent_id,title,abstract,claim,ipc,applicant,application_date
```

## 规则文件

复制 `rules.template.json`，只填入获准公开的术语。独立命中概念的格式如下：

```json
{
  "concept_id": "PUBLIC-CONCEPT-001",
  "category": "公开分类",
  "canonical_term": "公开概念名称",
  "variants": ["公开词组"],
  "match_policy": {"mode": "standalone"},
  "excluded_phrases": [],
  "public_source_ids": ["公开来源编号"]
}
```

依赖上下文的概念格式如下：

```json
{
  "concept_id": "PUBLIC-CONCEPT-002",
  "category": "公开分类",
  "canonical_term": "公开概念名称",
  "variants": ["需要语境消歧的词组"],
  "match_policy": {
    "mode": "cooccurrence",
    "required_any": ["PUBLIC-CONTEXT-001"]
  }
}
```

被引用的语境词必须出现在同一句中；若文本没有句界，则必须位于配置的字符窗口内。

## 运行

在仓库根目录运行：

```bash
python -m loop.step1.step1_public \
  --config loop/step1/config.example.json \
  --input path/to/patents.csv \
  --output-dir loop/step1/output
```

`result.csv` 包含 S/E 路由、Step 1 标签、纳入概率、逆概率权重和 JSON 审计字段。
`manifest.json` 中的 `"input_source"` 固定为空，只记录输入文件名和 SHA-256，不保存本地
路径或授权数据地址。

## 可复现抽样

对身份键为 `dataset_id|patent_id` 的 E 类记录，是否纳入由下式决定：

```text
u = uint64(SHA256(seed + "|" + identity)[0:8]) / 2^64
selected = u < e_sample_rate
```

输入、规则、seed 和抽样率相同时，入选的专利编号保持一致。
