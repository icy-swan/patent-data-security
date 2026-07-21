# Step 4 小分类模型改进方案（负例召回专项）

> 状态：**执行中**（2026-07-21 已获批准）。执行决策：
> ①**跳过阶段 0**（旧 RoBERTa 基线管道随 simplify 分支删除了冻结契约中间产物已跑不起来，
> 且其诊断结论已由实测 SFT `ep-20260720173842-4r782` 与 §1.6 反推确证——负类 recall 0.20）；
> ②**只走路线 A（方舟 seed-2-0-mini）**为最终交付，**不做本地 RoBERTa（路线 B）**；
> ③不装本地 torch，蒸馏数据直接喂方舟 LoRA SFT。
> 目标：训练一个**更强的小分类模型**（判别专利是否属于数据安全领域），
> 重点解决当前"正例几乎全对、负例大量误判"的问题。
> 选定路线：**教师蒸馏为主**（glm-5.2 作教师；其 step2 判定即现成蒸馏标签），
> 人工 4000 条转为**冻结金标准测试集**，不再全部用于训练。
>
> **阶段 1 已完成**（`pipeline/step4_distill/`）：从 step2 `result.csv` + `tasks.sqlite3`
> 挖出三档负例，重配比至 1:2（33.3% 正），输出 `data/step4_distill/`——
> 训练集 10,107 条（train 9,096 / val 1,011，均 33.3% 正）、冻结金标准测试集 4,000 条、
> 低置信待复核 pos 347 / neg 139。去泄漏：排除 4,000 金标准 + 5 条逐字文本重复，0 泄漏 0 重复。
>
> **阶段 1 增强（2026-07-21）**：①静态指令从 410 字扩为 ~1.9k 字——从 step2
> `scope.json` **动态注入** 13 条受控范围 + 5 条负例边界，与教师 glm-5.2 对齐（零泄漏、
> 推理时可得，仍远小于 step2 的 12.4k 字）；②按用户决策产出 **两套 SFT 供 A/B**：
> `sft_label/`（target=纯标签）与 `sft_reason/`（target=先理由后标签的 CoT 蒸馏，
> reason 取自 step2 教师判词）。两套各含 train/validation/train_full。校验：
> 标签一致性 0 冲突、reason 全非空、跨 split 0 重复；方舟 dry-run 均 `Succeed`
> （label 变体 ~19.7M tok，reason 变体 ~20.8M tok，10,082 行）。

---

## 1. 问题诊断（为什么现在负例判不准）

### 1.1 训练分布与部署分布严重错配
- 现有人工标注 `data/step3/result.csv`：正例 `true` 3046（76.2%）、负例 `false` 954（23.8%）。
- Step 2 机器判定的真实先验：正例 6718 / 21599 ≈ **31%**（实测），即真实世界近 7 成是负例。
- 结论：训练集 76% 正、部署 31% 正，模型决策边界被系统性推向"倾向正例"，
  一到真实数据（负例为主）就大量误报。观察到的"负例判错多"与此完全一致。
- **已跑 SFT（`ep-20260720173842-4r782`，seed-2-0-mini）在 400 条 held-out test 上的实测**：
  accuracy 0.81，但 **balanced_accuracy 仅 0.60**、**OTHER(负类) recall 0.20**、假阳性率 0.80、
  macro-F1 0.61（正类 recall 1.0、precision 0.80）。说明模型几乎"逢负必错"，
  0.81 的 accuracy 只是靠该测试集 76% 是正例撑起来的——**它主要学到了数据集偏置，而非判别边界**。

### 1.2 负例数量少且偏"硬"，但覆盖不足
- Step 3 负例定义为 `route=S AND step2=OTHER`（[sampling.py](file:///Users/bytedance/git/patent-data-security/pipeline/step3/sampling.py#L804-L809)），
  即通过了关键词初筛、"看起来像"数据安全但实际不是的边界样本。方向正确，
  但只有约 954 条，不足以让小模型学稳负类边界。
- 明确排除了 easy negative（`E → OTHER`），这对大模型 OK，但小模型仍需要一定量
  "明显负例"来锚定基本盘。

### 1.3 SFT 任务对小模型过重
- 现有 SFT assistant target 是**完整结构化 JSON**（label + confidence + scope +
  逐字 evidence + reason），见 [data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L399-L438)。
  evidence 还要求是原文逐字子串。
- 对大教师没问题，但**小模型容量有限**，被迫把大量容量用于结构化生成与证据抽取，
  真正的二分类决策（尤其微妙的负类边界）反而学不透。

### 1.4 训练与选择口径放大了不平衡
- 损失为**未加权交叉熵**，best checkpoint 以**验证集 accuracy** 选择
  （[train.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/train.py#L123-L124)、[train.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/train.py#L213)）。
  在不平衡数据上，accuracy 会奖励"多数类塌缩"。
- 好消息：[metrics.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/metrics.py) 其实**已经**计算
  balanced_accuracy、macro_f1、per-class P/R/F1、混淆矩阵、ROC-AUC、AP，
  只是没有被用作选择/主指标。改动成本低。

### 1.5 结论
- **再改 glm-5.2 提示词重跑，对小模型没有帮助**：瓶颈不在教师标注措辞，
  而在①学生数据分布 ②负例覆盖 ③任务格式 ④不平衡处理与评估口径。

### 1.6 上游基线：用 step3 人工标注反推 step1/step2 的真实准确率（实测）

把 [step3 result.csv](file:///Users/bytedance/git/patent-data-security/data/step3/result.csv)
的 4000 条 `human_evaluation` 作为**金标准**（正 3046 / 负 954），按 `patent_id`
关联回 [step1](file:///Users/bytedance/git/patent-data-security/data/step1/2021/result.csv)（关键词法，route S=判正 / E=判负）
与 [step2](file:///Users/bytedance/git/patent-data-security/data/step2/2021/result.csv)（glm-5.2 标注），
4000 条**全部命中**，实测如下。

> ⚠️ step3 是**分层抽样**：它几乎只抽了 route=S 的样本（S 3954 / E 46），
> 且没有抽 `route=E & OTHER`（易负例）这一大层。所以直接在 4000 条上算的"朴素口径"
> 会低估 step1、并让 step2 的负类看起来更难。下面同时给**朴素口径**（4000 条原样）与
> **总体重加权口径**（把各层按 step2 全量 21,599 的真实层大小加权，更接近真实分布）。

**step2（glm-5.2 标注）——这就是小模型要对齐/超越的教师上限**

| 口径 | accuracy | balanced_acc | macro-F1 | 正类 P/R | 负类 P/R |
|---|---|---|---|---|---|
| 朴素（4000 条） | 0.928 | 0.909 | 0.903 | 0.960 / 0.945 | 0.833 / 0.873 |
| 总体重加权（N≈21,599） | **0.964** | **0.954** | **0.958** | 0.959 / 0.927 | 0.966 / 0.981 |

> 关键点：**glm-5.2 教师本身就有约 macro-F1 0.90（朴素）～0.96（总体）的水平**，
> 负类在总体口径下 recall≈0.98。这说明：①教师标签质量足够做蒸馏；
> ②小模型的合理目标是**逼近教师**，而不是止步于此前 ≥0.88 的偏低目标。

**step1（关键词法，route S=判正）——高召回、低精度的粗筛器**

| 口径 | accuracy | balanced_acc | macro-F1 | 正类 P/R | 负类 P/R |
|---|---|---|---|---|---|
| 朴素（4000 条，≈99% 是 S，退化） | 0.755 | 0.499 | 0.439 | 0.761 / 0.988 | 0.196 / 0.009 |
| 总体重加权（N≈21,599） | 0.867 | 0.898 | 0.859 | 0.711 / **0.986** | 0.992 / 0.810 |

> 关键点：关键词法定位是**召回优先的初筛**——正类 recall≈0.99（几乎不漏），
> 但正类 precision 仅 0.71（约 3 成误召，即 2,781 条假阳性靠 step2 才被纠回 OTHER）。
> 朴素口径的 balanced_acc≈0.50 是抽样退化造成的假象，总体口径才是真实画像。

**对本方案的直接含义**
1. **教师可信**：glm-5.2 总体 macro-F1≈0.96，蒸馏标签质量有保障（低置信样本仍建议人工过一遍）。
2. **目标要抬高**：小模型不应满足于 macro-F1 0.88，应对齐教师（见第 2 节已上调目标）。
3. **难点在 `route=S & OTHER`**：这一层正是 step1 的假阳性（关键词命中但实为负），
   共 3,038 条，也是 step2 负类的主战场——与 4.1 "硬负例尽量全取"的策略完全吻合。

---

## 2. 改进目标（可度量）

**锚定教师上限**（见 1.6）：glm-5.2 在同分布金标准上 macro-F1 约 **0.90（朴素）/ 0.96（总体）**，
负类 recall 总体口径约 **0.98**。小模型作为蒸馏学生，目标是**尽量逼近教师**，而非此前偏低的 0.88。

以**部署分布（约 31% 正）**抽样的测试集为准，而非训练集分布：

| 指标 | 现状 seedmini（实测） | 教师 glm-5.2（总体） | 小模型目标 |
|---|---|---|---|
| 负类 recall（1−误报率） | 0.20（主要痛点） | ~0.98 | ≥ 0.90 |
| macro-F1 | 0.61 | ~0.96 | ≥ 0.90 |
| balanced accuracy | 0.60 | ~0.95 | ≥ 0.90 |
| 正类 recall | 1.00 | ~0.93 | 保持 ≥ 0.92（允许略降换负类） |

> 目标定在"逼近教师、留一点蒸馏损耗"的水平（macro-F1 ≥ 0.90）。
> 原始 accuracy 不再作为主指标，仅作参考。
> 若达到教师同级（macro-F1 ~0.95）则视为超预期。

---

## 2.5 方舟精调能力实测（2026-07-21，arkcli 实查，非记忆）

在拍板方案 2/3/4 前，用 `arkcli train finetune` 实查了方舟的精调能力，结论直接影响设计：

**可训练的小模型候选（均已实测支持 SFT + LoRA）**

| 模型 | 主版本 | 状态 | 支持训练方法 | dataset_schema |
|---|---|---|---|---|
| `doubao-seed-2-0-mini` | 260428 | **Published** | sft / lora / dpo / dpo-lora / grpo / grpo-lora / opd(-lora) | `ImageRecognitionSFT`(多模态 schema，兼容纯文本) |
| `doubao-seed-1-6-flash` | 250828 | Retiring | sft / lora / dpo(-lora) / grpo(-lora) / ppo / opd(-lora) | `ImageRecognitionSFT` |
| `doubao-1-5-lite-32k` | 250115 | Retiring | sft / lora / dpo(-lora) / grpo | `PromptResponse`(纯文本) |

> 推荐基座：**`doubao-seed-2-0-mini`（Published，能力最全，且是唯一非 Retiring 的小模型）**。
> `flash` / `lite-32k` 都是 Retiring 状态，不建议作为新训练的落点。

**训练耗时（关注点：效果 + 耗时，不看价格）**

以你 2026-07-20 实测为锚：**4000 条 SFT，约 3 分钟**跑完（日志 global_step 从 0 到 12、
avg_num_tokens≈970/step、约 12 步完成）。据此线性外推（同超参、同 seq_len 量级）：

| 训练样本数 | 预计训练耗时(≈) |
|---|---|
| 4 千（实测锚点） | ~3 分钟 |
| 2 万 | ~15 分钟 |
| 3 万 | ~23 分钟 |
| 5 万 | ~38 分钟 |
| 10 万 | ~75 分钟 |

> 结论：即便抽到 5–10 万样本，**单次 SFT 也就几十分钟级**，耗时完全可接受。
> 真正吃时间的是**排队/资源调度**与**教师蒸馏打标**（给几万条专利跑 glm-5.2 推理），
> 不是训练本身。因此策略上**放心把数据喂足**，用数据量换效果。
> LoRA 与全量 SFT 训练时长同量级；如需更快迭代可先 LoRA。

**部署能力（seed-2-0-mini 基座潜在能力，精调产物需再用 `--custom-model-id` 复核）**
- 支持 batch_inference / batch_inference_job / provisioned_throughput_unit_v2 / share_service。
- 我们是**离线批量给专利打标**的场景，`batch_inference` 正好命中，不必常驻 online endpoint。

**关键结论（纠正上一版方案的两个想当然）**
1. **方舟 SFT 平台的超参只有 epoch / batch_size / learning_rate / warmup_step_rate /
   seq_len / save_model_per_epoch / test_every_n_steps 等，__没有暴露 class weight 或
   focal loss__。** 因此"训练时加类别权重/focal"这条在方舟托管 SFT 上**做不到**，
   上一版方案 4.3 属于对 HuggingFace 本地训练的惯性套用，需要改。
2. **方舟原生支持 DPO 与 GRPO。** 这才是在方舟平台上治"负例判错"的正确武器——
   用偏好数据（正确判负 vs 错误判正）做偏好对齐，比在 SFT 里硬调权重更贴平台能力。

> 因此：**方舟托管小模型走"SFT 配比 + 偏好对齐(DPO)"路线**；
> 训练时"类别加权 / focal loss / 自定义选择指标"在方舟托管 SFT 上均不可用，
> 不平衡只能靠数据配比解决（本方案已据此设计 §4.1）。

---

## 3. 方案总览

**最终交付：路线 A —— 方舟托管小模型 `doubao-seed-2-0-mini`（生成式，可 batch 推理的 LoRA 精调产物）。**
已拍板不做本地 RoBERTa（路线 B），不装本地 torch，蒸馏数据直接喂方舟 LoRA SFT。
方舟平台不支持训练时类别加权，故不平衡完全在**数据侧**解决（配比 + 三档负例课程），
残余误报再用平台原生 **DPO** 偏好对齐治理。

四条改动线：

1. **数据线（最高优先级）**：教师蒸馏挖大量负例，重配比至接近部署先验（1:2，33.3% 正）。
2. **任务格式线**：把 SFT target 从重结构化 JSON 精简；产出 **A/B 两套变体**——
   `sft_label`（target=纯标签）与 `sft_reason`（target=先理由后标签的 CoT 蒸馏），
   用同一份数据、同一超参训练，在冻结金标准上比负类 recall / macro-F1 决出胜者。
3. **不平衡处理线**：SFT 靠数据配比（平台不暴露 class weight / focal）；残余误报用 **DPO**。
4. **评估线**：固定部署分布金标准测试集，报告混淆矩阵与 per-class 指标，两变体同口径可比。

---

## 4. 详细设计

### 4.1 数据线：教师蒸馏 + 负例扩充（核心）

**数据来源（实测规模，2021 一年）**
- 正例池：step2 `DATA_SECURITY` **6,718** 条（其中 route S 6,597 / route E 121）+ 人工确认正例。
- 负例池：step2 `OTHER` **14,881** 条，分两类——
  - **易负例 route E & OTHER：11,843 条**（明确不相关，用于锚定基本盘，取部分）。
  - **硬负例 route S & OTHER：3,038 条**（通过关键词初筛但判负的边界样本，小模型最缺，尽量全取）。
- 低置信可挖：step2 低置信正类（<0.85）**626** 条、低置信负类 **215** 条 —— glm-5.2 自身不确定、
  最接近决策边界，是难例主力，优先送人工/教师终审确认后进训练。
- 若纳入多年份（step1/2021 候选就有 **608,110** 条），负例池可扩到十万级。

**负例挑选策略（分三档，构造"课程"）**
1. **clean negative（明显负例）**：`route=E AND step2=OTHER` 且 glm-5.2 高置信（confidence ≥ 0.85）。
   量大（约 1.1 万+），给小模型锚定基本盘，取一部分即可。
2. **hard negative（边界负例）**：`route=S AND step2=OTHER`（约 3,038），
   即标题/摘要含"数据/安全/加密/区块链/隐私"等诱导词、通过关键词初筛但结论为 OTHER 的样本。
   **这是小模型最缺、最该补的，尽量全取。**
3. **low-confidence negative（低置信负例）**：glm-5.2 判 OTHER 但 confidence < 0.85（约 215）。
   最接近决策边界、信息量最高；**先送人工确认**，避免把教师的错标喂进去。

**标签口径（蒸馏教师）**
- 直接采用 glm-5.2 的 step2 判定作为蒸馏标签（正例 DATA_SECURITY / 负例 OTHER）。
- 低置信样本（<0.85，正负各 626 / 215）**先人工快速确认**再进训练，防止把教师错标喂给学生。
- 人工 4000 条中的**负例优先并入金标准测试集**（最可信），不进训练集。

**配比目标**
- 训练集正:负 **≈ 1:2**（贴近部署先验 31% 正）；若担心正类召回下降，可退一步 1:1。
- 规模目标：单 2021 年即可支撑 **2万～3万级**训练集（负例池 14,881 + 正例池 6,718 已足够按 1:2 配比取到约 2 万）；
  纳入多年份可上探 **5万～10万级**。样本数对小模型收益大、单次 SFT 也就几十分钟（见 2.5 耗时表），
  耗时不是瓶颈，**优先把负例（尤其硬负例 3,038 尽量全取）喂足**。

**去泄漏**
- 沿用现有跨集合逐字文本去重逻辑（[data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L366-L389)），
  确保蒸馏训练集与金标准测试集**无文本重叠**。

**产物（已实现，`pipeline/step4_distill/`）**
- 蒸馏数据构建脚本 [build.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/build.py)：
  从 step2 `result.csv`（route + label + confidence 分档）+ `tasks.sqlite3`（专利正文）
  读取，输出重配比后的数据集与 manifest（记录来源哈希、配比、三档负例数量、去泄漏）。
- 静态指令**动态注入**：`build_instruction()` 从 step2 `scope.json` 读入 13 条受控范围
  + 5 条负例边界，与教师 glm-5.2 对齐（零泄漏、推理时可得，~1.9k 字，远小于 step2 的 12.4k 字）。
- 输出 **A/B 两套 SFT 变体**（方舟 `ImageRecognitionSFT` schema，兼容纯文本）：
  - `data/step4_distill/sft_label/`：target=纯标签（`OTHER` / `DATA_SECURITY`）。
  - `data/step4_distill/sft_reason/`：target=先理由后标签的 CoT 蒸馏，reason 取自 step2 教师判词。
  - 每套含 `train.jsonl` / `validation.jsonl` / `train_full.jsonl`（后者供服务端
    `--validation-percentage` 切分，绕开 MaaS CLI 无法上传独立验证文件的限制）。
- 另导出 `classifier/*.jsonl`（保留字段级结构，备用），`gold_test.jsonl`（冻结金标准），
  `review/low_conf_*.jsonl`（低置信待人工复核）。
- **不覆盖**现有 `data/step4/dataset/`，独立放在 `data/step4_distill/`，保护原冻结契约。

### 4.2 任务格式线：为小模型减负（A/B 两变体）

把 assistant target 从现有**完整结构化 JSON**（label + confidence + scope + 逐字 evidence + reason，
见 [data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L399-L438)）大幅精简，
让小模型容量集中在判别本身。产出两套变体做 A/B：

- **`sft_label`（纯标签）**：assistant target 只有 `OTHER` / `DATA_SECURITY`。最轻、推理最快
  （`max_tokens` 极小），完全依赖数据配比修负类。
- **`sft_reason`（先理由后标签，CoT 蒸馏）**：target 为 `{"reason":"<一句话>","label":"..."}`，
  reason 取自 step2 教师判词。把教师判别逻辑迁移给学生，对微妙负类边界通常更有利；
  代价是每条多生成 ~250 token、`max_tokens` 需放大到 ~512。

两变体共用同一份蒸馏数据、同一超参，唯一变量是 target 形态，保证对比干净。

- 数据格式落到方舟 `ImageRecognitionSFT`（seed-2-0-mini 的 sft schema，兼容纯文本）。
  具体字段以 `arkcli models finetune-config <model> <version> --type sft` 返回的
  `dataset_schema` 为准，创建前用 `arkcli train finetune create --dry-run` 做服务端校验。
- **不直接复用**现有 [data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L399-L438)
  的 SFT 导出（重结构化 target），由 [build.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/build.py) 新写精简 target 导出器。

> 说明：现有 `prepare` 强校验 4000/500/500 且核对 Step-3 manifest 哈希
> （[data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L304-L322)、[data.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/data.py#L31)）。
> 蒸馏数据规模不同，**不要修改**现有 `prepare` 的契约；用新构建脚本/新目录承载，
> 保持原论文复现路径不被污染。

### 4.3 不平衡处理线：数据配比 + DPO（方舟能力实测后修订）

方舟托管 SFT **不支持训练时加权**，不平衡完全在数据侧解决，残余误报用 DPO 治理：

- **SFT 阶段靠数据配比**：方舟 SFT 超参里**没有 class weight / focal loss**
  （实测超参只有 epoch / batch_size / learning_rate / warmup_step_rate / seq_len /
  save_model_per_epoch / test_every_n_steps）。所以不平衡只能在**数据侧**解决——
  即 4.1 的负例扩充 + 重配比（正:负 ≈ 1:2，33.3% 正）。
- **DPO 偏好对齐治残余误报（方舟原生支持）**：SFT 之后，用一批
  **(chosen=正确判负, rejected=错误判正)** 的偏好对做 DPO，专门压 SFT 仍会误报的边界负例。
  偏好数据来源：SFT 初版模型在金标准上跑出的**假阳性样本**，配上教师给的正确负标签。
  这是方舟平台上比"加权"更对路的手段。
- LoRA vs 全量：默认 **LoRA**（迭代快、且方舟支持对 LoRA 产物部署；全量产物当前
  ArkCLI 不支持部署，需控制台）。

### 4.4 评估线：以部署分布为准（两变体同口径）

- 固定一个**部署分布测试集**（约 31% 正），可从 step2 全量分层抽样 + 教师/人工确认；
  **人工 4000 条中的负例优先并入金标准测试集**（它们最可信）。
- 报告：混淆矩阵 + per-class P/R/F1 + balanced accuracy + macro-F1 + ROC-AUC/AP
  （[metrics.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/metrics.py) 已全部支持）。
- 训练集分布口径的 accuracy 仅作附注，不作结论。
- 两个 SFT 变体（`sft_label` / `sft_reason`）的方舟产物都用 `batch_inference` 在
  **同一金标准测试集**上离线打标，套用同一套
  [metrics.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/metrics.py) 计算，保证 A/B 可比。

---

## 5. 执行阶段与验收

> 每个阶段可独立 review / 回滚。建议顺序执行，先拿数据线的收益。

### 阶段 1：教师蒸馏数据构建 ✅ 已完成
- 蒸馏构建脚本 [build.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4_distill/build.py)：
  从 step2 `result.csv`（route + label + confidence 分档）+ `tasks.sqlite3`（正文）挖三档负例
  （clean / hard / low-confidence），glm-5.2 判定即标签，输出重配比（1:2，33.3% 正）数据集 + manifest。
- 导出 **A/B 两套精简 target SFT**（`sft_label` / `sft_reason`），静态指令从 scope.json 动态注入。
- 验收（已达成）：train 10,107 条（1:2）、去泄漏 0 泄漏 0 重复、标签一致性 0 冲突、
  方舟 dry-run 均 `Succeed`；硬负例 spot-check 教师标签质量高。

### 阶段 2：路线 A 方舟 LoRA SFT（A/B 两变体）🔄 进行中
- 两变体各在 `doubao-seed-2-0-mini`（260428）上做 **LoRA SFT**，同超参
  （epoch3 / batch16 / seq32768 / lora_rank32），用 `train_full.jsonl` + 服务端
  `--validation-percentage 10` 切验证集。
  - `sft_label` → job `mcj-20260721153801-gqntb`
  - `sft_reason` → job `mcj-20260721153822-jqvdr`
- 训练耗时参考 2.5：~1 万条约十几到几十分钟，主要等排队/调度。
- 训练后两变体各导出 `batch_inference`，在冻结金标准 `gold_test.jsonl`(4000) 上打标，
  套 [metrics.py](file:///Users/bytedance/git/patent-data-security/pipeline/step4/metrics.py) 评估。
- 验收：负类 recall ≥ 0.90、macro-F1 ≥ 0.90（按 step3 分层权重加权）；A/B 择优为最终交付。

### 阶段 3（可选）：DPO 偏好对齐（训练约几十分钟）
- 若 SFT 胜出变体仍有残余误报：收集假阳性样本构造偏好对（chosen=正确判负 / rejected=错误判正），
  在方舟做 **DPO**，专治边界负例。
- 验收：负类 recall 相对阶段 2 进一步提升，且正类 recall 不明显回退。

### 阶段 4：复核与归档（0.5 天）
- 更新 [patent_identification_methodology.md](file:///Users/bytedance/git/patent-data-security/docs/patent_identification_methodology.md)
  第 9 节，记录蒸馏方案、配比、方舟精调（模型/方法/超参/训练耗时）、A/B 结论、指标口径变化。
- 保留原论文复现路径不变，蒸馏路径独立成节。

---

## 6. 不做什么（范围边界）
- **不**改 glm-5.2 提示词后重跑来"校验"——对小模型无效。
- **不**修改现有 `data/step4/dataset/` 的 4000/500/500 冻结契约与哈希校验。
- **不**把人工 4000 条全部用于训练——它们冻结为金标准测试集。
- **不**在方舟 SFT 上指望"训练时类别加权/focal"——平台不暴露该超参，不平衡走数据配比 + DPO。
- **不**做本地 RoBERTa（路线 B）、不装本地 torch——最终交付只走方舟路线 A。
- **不**用 Retiring 状态的 `flash`/`lite-32k` 作新训练落点，除非有明确理由。

---

## 7. 待确认项（review 时请拍板）

已定：**最终交付=方舟 seed-2-0-mini LoRA（路线 A）**；**配比=1:2（33.3% 正）**；
**任务格式=A/B 两变体（纯标签 vs 先理由后标签）由训练结果择优**。剩余待定：

1. **年份范围**：蒸馏数据是否只用 2021，还是纳入其它年份（需确认各年 step2 产物就绪）。
2. **低置信样本人工确认**：glm-5.2 低置信样本（正 347 / 负 139）是否人工过一遍后并入训练？
   （建议要，避免把教师错标喂给学生；量不大，成本可控。）
3. **金标准测试集分布**：当前冻结集为 76% 正（沿用 step3 分层抽样）；评估时是否需按 step2
   全量层大小加权换算到部署分布（约 31% 正）作为主口径。
