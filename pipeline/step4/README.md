# Step 4：RoBERTa 分类与 MaaS SFT 数据

从 Step 3 已冻结的 3,200/400/400 切分生成 Step 4 数据：

```bash
python -m pipeline.step4 prepare
```

命令只接受由 Step 3 根目录 `result.csv` 人工结果生成的 3,200/400/400 精简切分，严格检查
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

SFT 文件每行只有 `messages`，结构与提供的 MaaS 样例一致。只生成 train 和 validation；
测试集不导出为 SFT 文件。`index.csv` 用于把 MaaS 行号追溯到 `sample_id`，不要上传为训练集。

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
