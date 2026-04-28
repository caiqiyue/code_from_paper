# 四个数据集特征分析报告

**日期**: 2026-04-28  
**目的**: 分析 jobs / forums / microblog / congressional 四个数据集的特征分布，为算法性能差异提供数据支撑

---

## 1. 数据集基本信息总览

| 数据集 | Train 样本数 | Eval 样本数 | 数据类型 |
|--------|------------|------------|---------|
| **forums** | 10,000 | 1,000 | 列表 (str) |
| **jobs** | 10,000 | 10,000 | 列表 (str) |
| **microblog** | 10,000 | 10,000 | 列表 (str) |
| **congressional** | 257,680 | 28,632 | 列表 (str) |

---

## 2. 文本长度统计

### 2.1 Train 数据集长度统计

| 数据集 | 平均词数 | 中位数 | 标准差 | 最小 | 最大 | P25 | P75 | P90 | P95 |
|--------|---------|--------|--------|------|------|------|------|------|------|
| **forums** | 379.4 | 190 | 580.5 | 20 | 15,163 | 86 | 440 | 912 | 1,328 |
| **jobs** | 270.0 | 157 | 459.1 | 20 | 20,309 | 80 | 312 | 562 | 805 |
| **microblog** | 348.4 | 183 | 540.2 | 20 | 11,352 | 86 | 403 | 765 | 1,157 |
| **congressional** | 227.1 | 103 | 573.1 | 20 | 190,355 | 69 | 186 | 517 | 926 |

### 2.2 长度分布区间占比

| 长度区间 | forums | jobs | microblog | congressional |
|---------|--------|------|-----------|---------------|
| 0-50 | 11.4% | 9.3% | 10.1% | 13.3% |
| 50-100 | 17.5% | 23.2% | 19.5% | **34.3%** |
| 100-200 | 22.7% | 26.2% | 23.3% | 29.3% |
| 200-500 | **26.8%** | **28.9%** | **27.8%** | 12.8% |
| 500-1000 | 13.0% | 9.2% | 12.7% | 5.9% |
| 1000-5000 | 8.4% | 3.1% | 6.5% | 4.4% |
| 5000+ | 0.2% | 0.1% | 0.2% | 0.0% |

### 2.3 关键发现

1. **congressional 文本最短**：34.3% 集中在 50-100 词区间，中位数仅 103 词
2. **forums/jobs/microblog 分布相似**：都有约 25-29% 在 200-500 词区间
3. **forums 文本最长**：平均 379.4 词，显著高于其他数据集

---

## 3. 数据集领域特征

### 3.1 Forums

**样本示例**：
```
[0]: Sticky: Feeling Suicidal, PLEASE Dial 911 or Crisis Hotline!!! Symptoms on day 1 from anti dep?? How do I avoid overwhelming myself? article about depression links between Fathers and Daughters...
[2500]: On my startup list these two appear. However, in the apps listing they do not appear so I cannot just uninstall them. Looking around the web they look like maleware...
[5000]: One of our french users observes non-negligible differences between the stresses displayed at the nodes (here von Mises, averaged corner data in the figure below...
```

**特点**：
- 领域：在线论坛讨论（Reddit/StackOverflow 类型）
- 内容：技术问题、社会话题、健康咨询混杂
- 风格：非正式、多样化、长度差异大
- 问题类型：开放式问题、讨论帖、求助帖混合

### 3.2 Jobs

**样本示例**：
```
[0]: March 10 2019 until March 30 2020 with renewal contract possible. The Visa sponsorship that we provide takes about three months to process therefore all applications Must be sent in by December 15...
[2500]: Home Jobs Skills Map English graduate jobs – What are your options? In a world where a majority of non-tech jobs are focused on communication skills...
[7500]: Carter Bond Solicitors is a boutique legal firm based in Stanmore, North West London. We are young, dynamic and growing...
```

**特点**：
- 领域：招聘/求职信息
- 内容：职位描述、技能要求、薪资范围
- 风格：半正式、结构化
- 特点：长度相对较短且集中

### 3.3 Microblog

**样本示例**：
```
[0]: LG's just released its new smart watch. It is called LG Watch Urbane 2 LTE. The "LTE" tells you that this is a.. LG Watch Urbane 2 cancelled...
[2500]: • Remember that young people participate for their enjoyment and benefit, not yours. • Applaud good performance and efforts from all individuals and teams...
[5000]: As a taboo that most TV networks broach with utmost caution, school shootings almost seem like they would have long since come into play on Sunny...
```

**特点**：
- 领域：社交媒体评论
- 内容：产品评论、事件讨论、观点表达
- 风格：短文本为主，信息密度高
- 特点：与初始化数据（C4 新闻类）较接近

### 3.4 Congressional

**样本示例**：
```
[0]: I thank my hon. Friend for her comments on the need to transition as quickly as possible. Through this crisis we have seen the volatility that this country is exposed to because of our reliance on fos...
[64420]: My constituency boasts some of the nation's most energy-intensive industries, from manufacturing to being the home of UK brewing. As such, I welcome the news that there will be energy subsidies for en...
[128840]: I congratulate the hon. Member for Richmond Park (Sarah Olney) on securing this debate about the impact of aircraft noise on local communities...
```

**特点**：
- 领域：英国国会辩论记录
- 内容：政策辩论、立法讨论
- 风格：正式、程式化、用语规范
- 特点：文本最短、最结构化

---

## 4. Initialization 数据分析

**数据来源**: C4 English Web Text (约 87,000 篇)

| 统计项 | 值 |
|--------|-----|
| 样本数 | 87,000 |
| 平均词数 | 364.8 |
| 中位数 | 192 |
| 最小/最大 | 20 / 23,973 |

**样本示例**：
```
[0]: Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity...
[100]: Come join us in celebrating the release of Greenville writer (and Presbyterian College professor) Terry Barr's first book of essays...
[1000]: Hongdae is named after the area around Hongik University and that tells me everything I need to know...
```

**特点**：
- 来源：公开网页文本
- 领域：新闻、博客、广告混合
- 风格：半正式、长短不一
- 与 Jobs 领域较接近

---

## 5. 与 PrE-Text Screening 结果对比

### 5.1 性能表现

| 数据集 | PrE-Text | NS (当前算法) | 差异 | 样本复杂度 |
|--------|----------|---------------|------|-----------|
| jobs | 0.2732 | **0.2761** | **+0.0029** | 简单 |
| congressional | 0.2950 | **0.2970** | **+0.0020** | 简单 |
| forums | **0.2501** | 0.2471 | -0.0030 | **复杂** |
| microblog | **0.2763** | 0.2749 | -0.0014 | **复杂** |

### 5.2 数据集复杂度分析

| 特征 | Jobs | Congressional | Forums | Microblog |
|------|------|---------------|--------|-----------|
| 平均词数 | 270 | 227 | **379** | 348 |
| 中位数 | 157 | 103 | **190** | 183 |
| 长度标准差 | 459 | 573 | **580** | 540 |
| >500词占比 | 9.3% | 6.4% | **21.6%** | **19.4%** |
| 内容领域 | 招聘 | 政治辩论 | **混合论坛** | **社交媒体** |
| 文本结构 | 半结构化 | **高度结构化** | **非结构化** | 短文本 |

---

## 6. 关键结论

### 6.1 成功的两个数据集 (jobs / congressional)

**共同特点**：
1. **文本较短**：平均 227-270 词
2. **领域相对单一**：招聘、政治辩论
3. **结构化程度高**：职位描述/议会记录
4. **>500词长文本占比低**：9.3% / 6.4%

### 6.2 失败的两个数据集 (forums / microblog)

**共同特点**：
1. **文本较长**：平均 348-379 词
2. **领域混合**：论坛讨论/社交媒体
3. **结构化程度低**：开放式讨论、评论
4. **>500词长文本占比高**：21.6% / 19.4%

### 6.3 假设

当前 selector 的 scoring formula 可能更适合：
- **短文本、高结构化、单一领域**的数据
- 而不适合：
- **长文本、非结构化、多样化领域**的数据

原因可能是：
1. **embedding 表征**：长文本的 embedding 可能被短文本主导
2. **genericity penalty**：长文本的 genericity 计算可能不准确
3. **候选选择偏差**：非结构化数据中，"好"样本的定义更模糊

---

## 7. 建议的后续方向

### 7.1 针对 Forums/Microblog 的改进

1. **候选生成策略**：
   - 增加候选数量（当前 24 initial + 32 generated = 56）
   - 针对长文本的 prompt 模板优化

2. **Selection 机制**：
   - 考虑文本长度作为选择因素之一
   - 对长文本使用不同的 genericity gate

3. **Bootstrap 策略**：
   - 调整 max_tokens 参数（当前 85，可能不适合生成长文本）

### 7.2 验证实验设计

| 实验 | 目标 | 参数调整 |
|------|------|---------|
| F1 | 验证 max_tokens 影响 | max_tokens: 85 → 150 |
| F2 | 验证候选数量影响 | candidate_count: 24 → 48 |
| F3 | 验证 initialization 来源 | 换成 forums 相关初始化 |

---

## 8. 附录：完整数据

### 8.1 Eval 数据统计

| 数据集 | Eval 样本数 | 平均词数 | 中位数 | 最小/最大 |
|--------|-----------|---------|--------|----------|
| forums | 1,000 | 376.5 | 185 | 20/4,974 |
| congressional | 28,632 | ~226 | - | 20/12,743 |

### 8.2 各数据集完整长度分布

**Forums (n=10,000)**:
```
0-50:      1144 (11.4%)
50-100:    1754 (17.5%)
100-200:   2269 (22.7%)
200-500:   2675 (26.8%)
500-1000:  1303 (13.0%)
1000-5000:  836 (8.4%)
5000+:        19 (0.2%)
```

**Jobs (n=10,000)**:
```
0-50:       931 (9.3%)
50-100:    2324 (23.2%)
100-200:   2621 (26.2%)
200-500:   2887 (28.9%)
500-1000:   915 (9.2%)
1000-5000:  312 (3.1%)
5000+:        9 (0.1%)
```

**Microblog (n=10,000)**:
```
0-50:      1008 (10.1%)
50-100:    1952 (19.5%)
100-200:   2328 (23.3%)
200-500:   2780 (27.8%)
500-1000:  1269 (12.7%)
1000-5000:  646 (6.5%)
5000+:        17 (0.2%)
```

**Congressional (n=257,680)**:
```
0-50:      34164 (13.3%)
50-100:    88308 (34.3%)
100-200:   75445 (29.3%)
200-500:   32938 (12.8%)
500-1000:  15262 (5.9%)
1000-5000: 11388 (4.4%)
5000+:        170 (0.1%)
```