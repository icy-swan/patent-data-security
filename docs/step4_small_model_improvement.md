# Step 4 小分类模型改进方案（数据安全二分类）

> 状态：**已交付**（2026-07-23）。最终方案：方舟托管 `doubao-seed-2-0-mini` LoRA SFT，
> 真实分布数据 + 纯字符串 prompt + sknwc 超参。
> 目标：训练一个更强的小分类模型（判别专利是否属于数据安全领域），
> 解决此前"正例几乎全对、负例大量误判"的问题。

---

## 1. 问题与目标

**痛点**：早期 SFT（`ep-20260720173842-4r782`）在 held-out 上 accuracy 0.81 但
balanced_accuracy 仅 0.60、负类 recall 0.20——模型学到的是数据集偏置而非判别边界。
根因是训练集 76% 正、部署约 31% 正的分布错配。

**基线（金标准 10,000 条口径，见 [step3_baseline_accuracy.md](file:///Users/bytedance/git/patent-data-security/docs/step3_baseline_accuracy.md)）**

| 上游 | accuracy | macro-F1 | 正类 recall | 负类 recall |
|---|---|---|---|---|
| step1 关键词法 | 0.675 | 0.648 | 0.985 | 0.384 |
| step2 GLM-5.2（教师） | 0.965 | 0.965 | 0.980 | 0.951 |

**目标**：小模型作为蒸馏学生逼近教师 GLM-5.2，macro-F1 ≥ 0.90、负类 recall ≥ 0.90。

---

## 2. 最终方案

**交付**：方舟 `doubao-seed-2-0-mini`（260428）LoRA SFT，纯文本二分类，在线接入点推理。
不平衡完全在**数据侧**解决（方舟托管 SFT 不暴露 class weight / focal loss）。

### 2.1 数据构建（[build_manual.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/build_manual.py)）

以人工金标准 [result.csv](file:///Users/bytedance/git/patent-data-security/data/step3/result.csv)
（10,000 条，`human_review_label`，DATA_SECURITY 4844 / OTHER 5156）为核心：

1. **金标准 50/50 分层** → 5,000 条**冻结为测试集**（自然比例，不进训练）+ 5,000 条训练池。
2. **训练集 6,813 条**：全部训练池正例 + 金标准/step2 负例补足，压到 **31% 正**（贴合线上先验）。
   - 正 2,112 / 负 4,701；负例 = 金标准 1,887 + step2 教师 2,814。
3. **验证集 1,000 条**：从训练池抽，同样 31% 正（正 310 / 负 690）。
4. **去泄漏**：title+abstract+claim 拼接取 SHA256 作 text_key，train / val / test 三者互斥
   （挡近重复专利：同文本不同 patent_id）。

**指令**：`build_instruction(with_reason=False)`，1,883 字，从 step2 `scope.json`
动态注入 13 条受控范围 + 5 条负例边界，与教师对齐。**Reason 字段仅用于历史 CoT 实验，
本方案 target 为纯标签，指令/prompt 中不含 reason。**

### 2.2 数据格式（关键：纯字符串，非 parts 列表）

SFT 每条为 `{"messages":[{"role":"user","content":<纯字符串>},{"role":"assistant","content":<纯标签>}]}`：

- **user.content 必须是纯字符串** = 指令 + 专利字段 JSON（`{title,abstract,claim,ipc,main_ipc}`）。
- assistant.content = 纯标签（`DATA_SECURITY` / `OTHER`）。

> ⚠️ **踩坑记录（务必遵守）**：早期照搬官方图片精调样本
> `SFT_ImageRecognition_Sample.jsonl` 把 user.content 写成 parts 列表
> `[{"type":"text","text":...}]`，导致精调后模型走多模态/thinking 链路，
> 推理时吐 `<think_never_used_...>` 保留 token 死循环（正例输出空、负例吐乱码串）。
> doubao-seed-2-0-mini 是文本模型，**user.content 一律用纯字符串**。
> 能用的历史任务 `mcj-20260720160144-sknwc` 正是纯字符串格式，据此定位并修复。

### 2.3 超参（沿用 sknwc）

| 参数 | 值 |
|---|---|
| 训练方法 | LoRA SFT |
| epoch | 2 |
| batch_size | 16 |
| learning_rate | 1e-5 |
| warmup_step_rate | 0.05 |
| lora_alpha | 4 |
| lora_rank | 32 |
| seq_len | 32768 |
| dyn_bsz | true |
| freeze_vit | true |

> **超参对照结论**：用旧超参（lr=1e-4 / alpha=16 / dyn_bsz=false）在同一份修正数据上重训
> （job `mcj-20260723110333-hdlpw`），eval loss 从 0.09 一路发散到 ~1.95、grad_norm 常年
> 数百——**lr=1e-4 对本任务过大直接训崩**。sknwc 的 lr=1e-5 才是正确落点。
> 早期把乱码归因到 seq_len/alpha 是误判：sknwc 用同样的 seq_len=32768 却正常。

---

## 3. 交付结果

**训练 job** `mcj-20260723110307-sz88g`（Completed）→ 自定义模型 `cm-20260723113829-9k8rx`
→ **接入点 `ep-20260723153741-fp9zh`（Running）**。

训练曲线健康：train loss 0.086 → 0.027，eval loss 0.090 → 0.036（单调下降，收敛好）。

**全量评估**：冻结金标准 [gold_test.jsonl](file:///Users/bytedance/git/patent-data-security/data/step4_manual/gold_test.jsonl)
5,000 条 · parsed 5000 / failed 0 · **0 乱码**
（脚本 [evaluate.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/evaluate.py)，产物 `data/step4_manual/eval_A_sknwc/`）。

### 口径1：自然比例（48.4% 正，金标原始）

| 指标 | 值 |
|---|---|
| **accuracy** | **0.9372** |
| **macro-F1** | **0.9372** |
| balanced-acc | 0.9380 |
| DATA_SECURITY | P=0.911 R=0.965 F1=0.937 |
| OTHER（负类） | P=0.965 R=0.911 F1=0.937 |

### 口径2：线上 31% 加权（真实先验）

| 指标 | 值 |
|---|---|
| **accuracy** | **0.9278** |
| **macro-F1** | **0.9190** |
| DATA_SECURITY | P=0.830 R=0.965 F1=0.892 |
| OTHER（负类） | P=0.983 R=0.911 F1=0.946 |

**混淆矩阵（自然比例）**：TN=2349 / FP=229 / FN=85 / TP=2337。

**结论**：
- 两类均衡无偏科，负类 recall 0.911（对比早期 0.20，痛点解决）。
- 31% 加权下负类 precision 0.983——线上判为 OTHER 的基本都对，过滤噪声价值高。
- 主要误差是 FP=229（OTHER 误判为 DATA_SECURITY），模型偏"宁可多报"。
- 达到教师 GLM-5.2（0.965）约 97% 水平，轻量本地可控，蒸馏目标达成。

---

## 4. 待办 / 可选优化

1. **误分类分析**：229 个 FP 中区分"模型错"与"金标本身可争议的边界样本"。
2. **压 FP**：若要提正类 precision，考虑负样本增强或调数据配比；平台原生 DPO 可做偏好对齐
   （chosen=正确判负 / rejected=错误判正），治残余误报。
3. **年份范围**：当前仅 2021，可评估纳入多年份扩充训练数据。

---

## 5. 关键产物索引

| 产物 | 路径 |
|---|---|
| 数据构建脚本 | [build_manual.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/build_manual.py) |
| 评估脚本 | [evaluate.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/evaluate.py) |
| 训练集 / 验证集 / 冻结测试集 | `data/step4_manual/{train,validation,gold_test}.jsonl` |
| 数据 manifest（哈希/配比/去泄漏） | [manifest.json](file:///Users/bytedance/git/patent-data-security/data/step4_manual/manifest.json) |
| 评估结果 | `data/step4_manual/eval_A_sknwc/`（summary + predictions + 误分类 CSV） |
| 基线报告 | [step3_baseline_accuracy.md](file:///Users/bytedance/git/patent-data-security/docs/step3_baseline_accuracy.md) |
