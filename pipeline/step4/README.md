# Step 4：RoBERTa 分类与 MaaS SFT 数据

从 Step 3 已冻结的 8,000/1,000/1,000 切分生成 Step 4 数据：

```bash
python -m pipeline.step4 prepare
```

命令只接受由 Step 3 根目录 `result.csv` 人工结果生成的 8,000/1,000/1,000 结构化切分，严格检查
结果文件与三个切分逐字段一致、唯一专利、`human_evaluation=true/false`，以及完全相同文本的
跨集合泄漏。Codex `simulation.csv` 不能作为 Step 4 输入。

输出结构：

```text
data/step4/
├── dataset/
│   ├── manifest.json
│   ├── classifier/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── sft/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── index.csv
├── model/roberta/
├── state/roberta/
└── reports/roberta/
```

SFT 文件每行顶层只有 `messages`。消息严格复用 Step 2 生产请求：system 使用同一法律文本、
受控范围、负向边界、分析维度和 JSON Schema，user 使用同一动态专利载荷，assistant 输出
Step 2 兼容的结构化分类结果（含标签、理由和逐字证据）。只生成 train 和 validation；测试集
不导出为 SFT 文件。`index.csv` 用于把 MaaS 行号追溯到 `sample_id`，不要上传为训练集。

历史实验使用的 4,000 条简化 Prompt 基线保存在
`data/step4/archive/data-security-binary-v1.0.0/`；它不是当前切分，不会与新的 canonical
`data/step4/dataset/` 混用。

安装并训练论文式 RoBERTa 分类器：

```bash
python -m pip install -e '.[step4]'

python -m pipeline.step4 train-roberta \
  --output-dir data/step4 \
  --model hfl/chinese-roberta-wwm-ext \
  --text-fields abstract \
  --epochs 4
```

默认只使用摘要，与参考论文一致。训练使用普通未加权交叉熵，每个 epoch 保存候选
checkpoint，以验证集 accuracy 选择最佳 checkpoint，再用 Softmax argmax 直接计算 train、
validation 和 test accuracy；不做类别加权、阈值校准或另一套稳健性模型。最终模型写入
`model/roberta/`，checkpoint 写入 `state/roberta/`，指标和逐条预测写入
`reports/roberta/`。

参考论文没有披露基座 checkpoint、epoch、学习率、batch size、weight decay、warmup 或随机
种子，因此这些是可复现的本项目实现参数，不应在论文中表述为原文参数。

如果显存有限，可调小 `--train-batch-size` 并增大 `--gradient-accumulation-steps`；支持
`--fp16`、`--bf16` 和 `--gradient-checkpointing`。本项目不实现或运行 SFT，生成的 JSONL
由用户在 MaaS 平台完成训练。
