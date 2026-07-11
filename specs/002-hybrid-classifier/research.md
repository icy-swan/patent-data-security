# Spec 002 Prompt 研究依据

## 1. 法律与行政法规

### 《中华人民共和国数据安全法》

- 来源：工业和信息化部转载的法律全文：
  <https://www.miit.gov.cn/zwgk/zcwj/flfg/art/2022/art_284b390b84484f10b0e43eeafaad0f6d.html>
- 关键条款：第三条、第二十一条、第二十七至三十二条。
- 对 Prompt 的贡献：
  - 数据处理覆盖收集、存储、使用、加工、传输、提供、公开等活动；
  - 数据安全同时要求有效保护、合法利用和持续安全能力；
  - 风险包括篡改、破坏、泄露、非法获取和非法使用；
  - 判断对象不应局限于静态数据，还应覆盖全流程处理活动。

### 《中华人民共和国个人信息保护法》

- 来源：中国人大网法律全文：
  <https://www.npc.gov.cn/WZWSREL25wYy9jMi9jMzA4MzQvMjAyMTA4L3QyMDIxMDgyMF8zMTMwODguaHRtbD9yZWY9aW1i>
- 关键条款：第四至九条、第五十一条。
- 对 Prompt 的贡献：
  - 个人信息是与已识别或可识别自然人有关的信息，匿名化后的信息除外；
  - 个人信息处理还包括删除；
  - 安全不仅是保密，还包括合法、正当、必要、目的明确、影响最小；
  - 技术措施应对应未授权访问、泄露、篡改和丢失等具体风险。

### 《网络数据安全管理条例》

- 来源：中国政府网全文：
  <https://app.www.gov.cn/govdata/gov/202409/30/520076/article.html>
- 关键条款：第九条、第六十二条。
- 对 Prompt 的贡献：
  - 网络数据是通过网络处理和产生的电子数据；
  - 网络数据处理覆盖收集至删除的完整活动链；
  - 保护措施是否与防篡改、破坏、泄露、非法获取、非法利用存在直接因果关系，是分类的重要证据。

## 2. 标准与论文

### GB/T 37988-2019《信息安全技术 数据安全能力成熟度模型》

- 来源：全国标准信息公共服务平台：
  <https://std.samr.gov.cn/gb/search/gbDetailed?id=91890A0DA63380C6E05397BE0A0A065D>
- 对 Prompt 的贡献：将数据安全理解为覆盖组织与技术、贯穿数据生命周期的持续能力，而非某个孤立术语。

### NIST FIPS 199

- 来源：NIST 官方页面：<https://csrc.nist.gov/pubs/fips/199/final>
- 对 Prompt 的贡献：以保密性、完整性和可用性描述信息安全属性，并把未授权访问、披露、中断、修改和破坏与安全影响连接起来。

### Saltzer & Schroeder (1975)

- 论文：*The Protection of Information in Computer Systems*，DOI: 10.1109/PROC.1975.9939。
- 作者出版物页：<https://web.mit.edu/Saltzer/www/publications/pubs.html>
- 对 Prompt 的贡献：信息保护判断应围绕受保护对象、授权边界、保护机制及其实际约束，而不是根据组件名称贴标签。

### Dwork, McSherry, Nissim & Smith (2006)

- 论文：*Calibrating Noise to Sensitivity in Private Data Analysis*。
- 原文：<https://www.iacr.org/archive/tcc2006/38760266/38760266.pdf>
- 对 Prompt 的贡献：隐私保护技术的相关性来自可说明的隐私保证与机制，而不是文本出现“隐私”或某种算法名称。

### Goldreich, Micali & Wigderson (1987)

- 论文：*How to Play Any Mental Game*。
- 作者页面与原文：<https://www.math.ias.edu/avi/node/843>
- 对 Prompt 的贡献：安全计算的关键是协议在明确参与方与威胁条件下限制信息泄露；技术名称本身不能证明某件专利以数据安全为核心。

## 3. 综合形成的判定框架

以下框架是基于上述法律、标准和论文作出的研究方法推导，不是任何单一来源的原文分类：

1. `A 保护对象/处理活动`：专利是否明确涉及数据、个人信息或法定数据处理活动；
2. `B 安全目标/风险/合规约束`：是否指向有效保护、合法利用、持续安全或具体风险；
3. `C 技术机制`：主权项必要特征是否直接作用于 A；
4. `D 因果与中心性`：C 是否直接产生 B，并构成发明核心而非附带效果。

- A-B-C-D 完整、具体且一致时归入类别 1；
- 存在实质性正向证据，但关键环节或中心性仍不确定时归入类别 2；
- 不满足前两项时归入类别 3“其他”。

类别 2 是人工核验层，不是“安全但非数据安全”层，也不是所有低置信样本的收容层。
