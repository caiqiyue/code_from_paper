# 2026-04-26 Round3 vs PrE-Text and Prior Tuning Analysis

## 1. 分析目的

本文把以下三类结果放在一起分析：

- `PrE-Text` 与旧版创新算法的原始 screening 对比结果  
  来源：[2026-04-24-pretext-screening-results.md](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/docs/2026-04-24-pretext-screening-results.md)
- round1 / round2 的纯参数微调结果  
  来源：[2026-04-26-stage1-parameter-tuning-screening-results-full.md](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/docs/2026-04-26-stage1-parameter-tuning-screening-results-full.md)
- round3 的结构微调结果  
  来源：[2026-04-26-round3-genericity-reference-smoothing-results.md](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/docs/2026-04-26-round3-genericity-reference-smoothing-results.md)

目标不是简单找“哪组分数最高”，而是回答四个问题：

1. `Stage 1` 当前到底哪一部分已经得到改善  
2. 哪一部分在 round1 / round2 / round3 之后仍然没有解决  
3. round3 的结构微调为什么会产生部分改善  
4. 下一步更应该继续调参数，还是已经应该进入下一层结构改造

## 2. 统一参照

先固定四个最重要的参照点。

### 2.1 `PrE-Text` 基线

| dataset | `PrE-Text` best_top1 |
| --- | ---: |
| `jobs` | 0.2731984829329962 |
| `congressional` | 0.2949640287769784 |
| `forums` | 0.25014487154722814 |
| `microblog` | 0.2762705387848682 |

### 2.2 旧版创新算法基线 `NS-S-*`

| dataset | old innovation best_top1 |
| --- | ---: |
| `jobs` | 0.2761061946902655 |
| `congressional` | 0.2969732322250308 |
| `forums` | 0.2470542785396948 |
| `microblog` | 0.27493312953763854 |

从原始 screening 现象看，旧版创新算法的基本问题已经很清楚：

- `jobs` / `congressional` 有正向信号
- `forums` / `microblog` 有负向信号
- 问题不是“算法完全无效”，而是“跨数据集不稳”

## 3. round1 / round2 说明了什么

## 3.1 已经被证实有效的方向

从两轮参数微调结果看，下面几个方向是有稳定信号的。

### 3.1.1 降低 `length_lambda`

`A2: length_lambda 0.20 -> 0.10`

效果：

- `jobs`: 0.279519595448799，优于旧基线
- `congressional`: 0.2964547281094044，接近旧基线
- `forums`: 0.24892151181507952，明显高于旧基线 0.2470542785396948
- `microblog`: 0.27709845879505796，超过 `PrE-Text`

这说明：

- 原来 `importance prior` 里的长度项对短文本和松散文本确实偏强
- 适度放松长度约束，对 `forums` / `microblog` 有帮助
- 而且没有像极端改 `support` 那样明显破坏强集

### 3.1.2 轻微降低 `lambda_generic`

`B1: lambda_generic 0.35 -> 0.30`

效果：

- `jobs`: 0.28053097345132744，round1/2 全部实验中最优
- `microblog`: 0.27900904343395744，round1/2 全部实验中最优
- `forums`: 0.24744060266563647，略高于旧基线
- `congressional`: 0.2913344999675935，明显下降

这说明：

- `genericity penalty` 的确在压制 `microblog`
- 但“全局降低 `lambda_generic`”会伤 `congressional`
- 所以问题不是“这项没用”，而是“这项当前作用方式太硬”

### 3.1.3 增大 `reference_top_k`

`E4: reference_top_k 4 -> 6`

效果：

- `forums`: 0.2494366106496684，round1/2 中最优
- `microblog`: 0.2755063049293084，接近但仍不算突出
- `jobs`: 0.2757269279393173，略降
- `congressional`: 0.2948992157625251，基本回到 `PrE-Text` 水平

这说明：

- `forums` 的关键问题之一确实在 `genericity` 参考邻域太窄
- 仅仅把 `reference_top_k` 放宽，就已经能稳定拉升 `forums`
- 但 simple mean 的放宽还不够，提升幅度有限

### 3.1.4 轻微重平衡 `density / novelty`

`E5: density_lambda 0.50 -> 0.45; novelty_lambda 0.30 -> 0.35`

效果：

- `jobs`: 0.27945638432364095
- `congressional`: 0.29548253289260484
- `forums`: 0.24808447620887258
- `microblog`: 0.2772895172589479

这是两轮纯参数微调里最接近“跨数据集稳健”的一组之一。

这说明：

- `importance prior` 里原先的密度偏好确实略强
- 但它不是当前最核心的单点故障
- 它更像一个次级修补方向，而不是主问题来源

## 3.2 两轮参数微调后仍然没解决的事

两轮结果最重要的负面结论是：

### 3.2.1 没有任何一组静态参数能让四个数据集同时明显上升

表现最接近稳健的 `A2 / E5 / E4` 也没有做到：

- `jobs` 稳
- `congressional` 稳
- `forums` 反超 `PrE-Text`
- `microblog` 稳定反超 `PrE-Text`

### 3.2.2 `forums` 一直没有被真正修好

即使 round1 / round2 里最好的 `E4`，`forums` 也只有：

- `0.2494366106496684`

仍低于 `PrE-Text`：

- `0.25014487154722814`

这说明：

- `forums` 不是靠简单调一个权重就能修复
- 它更像是 `genericity` 参考方式本身存在建模问题

### 3.2.3 `congressional` 对粗暴调参非常敏感

比如：

- `B1/B2` 一旦全局减弱 `genericity`
- `congressional` 就比旧创新基线明显更差

这说明：

- `genericity penalty` 这项对正式文本数据集确实有真实作用
- 所以不能简单删掉，也不能只靠全局降权解决问题

## 4. round3 结构微调到底改善了什么

round3 的核心不是再调一轮静态权重，而是把：

- `genericity reference = top-k simple mean`

改成：

- `genericity reference = top-k rank-weighted mean`

再配合不同的 `k` 和不同的 reference weight tail，构造了 `f1 / f2 / f3 / f4`。

这一步的本质是：

- 不再只问“genericity 要不要弱一点”
- 而是改成问“genericity 应该如何更平滑地参考公共分布”

## 4.1 round3 内部最优组

按 `best_top1` 看：

- `jobs` 最优：`f1 = 0.27920353982300883`
- `congressional` 最优：`f3 = 0.2969732322250308`
- `forums` 最优：`f2 = 0.24834202562616703`
- `microblog` 最优：`f2 = 0.27900904343395744`

## 4.2 明确改善的部分

### 4.2.1 `microblog` 已经被明显修复

旧创新基线：

- `NS-S-MICRO = 0.27493312953763854`

`PrE-Text`：

- `SP-S-MICRO = 0.2762705387848682`

round3 最优：

- `f2 = 0.27900904343395744`

这说明：

- 微调后的 `genericity reference smoothing` 的确在减轻 `microblog` 上的误罚
- 尤其是 `f2` 这种“保持 `k=6`，但让尾部 reference 权重衰减更快”的版本最有效
- 这与之前 `B1/B2` 的现象是连贯的：`microblog` 的核心问题就是 `genericity` 过强或过硬

### 4.2.2 `jobs` 和 `congressional` 没被破坏

旧创新基线：

- `jobs = 0.2761061946902655`
- `congressional = 0.2969732322250308`

round3：

- `jobs` 的 `f1/f3` 仍然高于旧基线
- `congressional` 的 `f3` 直接等于旧基线最优值

这说明：

- 把 `genericity reference` 平滑化，不像全局降 `lambda_generic` 那样会直接打掉正式数据集上的优势
- 结构微调比纯静态降权更保守，也更符合你的约束

### 4.2.3 已经证明“问题更像参考方式，而不是 penalty 存废”

round1 / round2 告诉我们：

- 直接降 `lambda_generic` 可以救 `microblog`
- 但会伤 `congressional`

round3 告诉我们：

- 不降整项 penalty
- 只改 `reference` 的聚合方式
- 也能救 `microblog`
- 同时不明显伤 `jobs` / `congressional`

这基本就把问题定位得很清楚了：

> 当前 `genericity` 的主要问题不是“这项不该有”，而是“这项参考公共分布的方式不够平滑”

## 5. round3 后仍然存在的问题

### 5.1 `forums` 还是没有真正修好

旧创新基线：

- `0.2470542785396948`

`PrE-Text`：

- `0.25014487154722814`

round3 最优：

- `f2 = 0.24834202562616703`

比旧基线高：

- `+0.00128774708647223`

但比 `PrE-Text` 仍低：

- `-0.00180284592106111`

这说明：

- round3 的结构改法对 `forums` 有帮助
- 但帮助幅度不够大
- `forums` 的问题只靠“平滑 reference 聚合”还没完全解决

### 5.2 `f3` 说明“单纯加宽邻域”不是答案

`f3` 是：

- `reference_top_k: 6 -> 8`
- 更宽的尾部权重

结果：

- `congressional` 最好
- `forums` 反而比 `f2` 更差
- `microblog` 也不如 `f2`

这说明：

- 不是 reference 越宽越好
- `forums` / `microblog` 需要的是“更平滑但仍局部优先”的参考
- 而不是简单把 reference 半径继续放大

### 5.3 `f4` 说明把旧稳健参数直接叠加进结构改动，并不会自动更强

`f4` 本来是：

- `f1` 的 weighted genericity smoothing
- 再叠加 `A2/E5` 的稳健参数组

但结果并不理想：

- `jobs` 下降
- `microblog` 下降
- `forums` 也没有回升

这说明：

- round1 / round2 中看起来“各自有帮助”的参数
- 到 round3 结构改动后并不一定还能继续叠加增益
- 也就是说，当前结构和静态参数之间已经开始出现耦合

这进一步支持一个判断：

> 继续做大范围静态参数拼装，收益会越来越低，甚至会掩盖真正的结构问题

## 6. 现在已经可以比较明确的机制判断

结合原始 screening、round1/round2、round3 三层结果，当前最可信的判断是：

### 6.1 已经被改善的部分

1. `genericity` 对 `microblog` 的误罚，已经通过 round3 的 reference smoothing 得到明显缓解  
2. `jobs` / `congressional` 上原有优势，在 round3 结构改法下基本被保住  
3. `genericity` 的问题来源，已经从“强度过大”收敛成“参考方式过硬”

### 6.2 仍然存在问题的部分

1. `forums` 仍未反超 `PrE-Text`  
2. `forums` 的问题不只是 `genericity` 过硬，可能还叠加了更深一层的候选分布或 boundary 形状问题  
3. 静态参数和结构改法已经开始出现耦合，继续大面积调权重的边际收益很低

## 7. 下一步应该怎么做

## 7.1 不建议做的事

### 7.1.1 不建议再做第三轮大范围静态调参

原因：

- round1 / round2 已经把主要静态参数方向基本扫过
- round3 又验证了结构微调比纯降权更有效
- 继续全局调静态参数，大概率只是在不同数据集之间换输赢

### 7.1.2 不建议直接放大实验规模

原因：

- `forums` 这个关键弱点还没修掉
- 现在去做更大的 formal 只会更贵地确认“不够稳”

## 7.2 建议优先进入的下一层结构改造

最合理的下一步不是回到大范围调参，而是继续沿着 `genericity` 往下改，但改得更具体。

### 7.2.1 第一优先：改 `genericity penalty` 的生效方式

当前 round3 只改了 reference 聚合方式，没有改 penalty 的作用形态。

下一步最值得尝试的是：

- 保留 weighted reference smoothing
- 再把 `genericity penalty` 改成更条件化的生效方式

也就是从：

- 所有候选都按同样公式吃同类惩罚

逐步改成：

- 只有更像“模板化公共模式”的候选才更强受罚
- 对于只是口语化、松散、但并不真正模板化的候选，惩罚更弱

这是因为：

- `microblog` 已经说明 simple smoothing 就能受益
- `forums` 仍然不够，说明还需要进一步减少误罚

### 7.2.2 第二优先：只在 `genericity` 之后，再看 `boundary / support` 的结构

如果下一步做完“条件化 genericity”后，`forums` 仍然没有被拉起来，那么才更有理由怀疑：

- `Stage 1` 候选本身分布有偏
- 或 `support / boundary` 选择机制对 `forums` 的长尾结构覆盖仍然不够

但在 round3 之后，我认为这个还不是第一优先级。

## 8. 结论

一句话结论可以写成：

> round3 已经证明，`genericity reference smoothing` 确实修复了当前创新算法在 `microblog` 上的主要问题，并且没有破坏 `jobs / congressional`；但 `forums` 仍未反超 `PrE-Text`，说明问题已经从“静态参数不合适”收敛为“`genericity penalty` 的参考方式已改善，但生效方式仍然过硬”，所以下一步应优先继续改 `genericity` 的结构，而不是再做大范围静态调参。

