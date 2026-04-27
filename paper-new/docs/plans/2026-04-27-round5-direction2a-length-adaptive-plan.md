# Round 5 方向 2a 实施计划：长度自适应惩罚（r1/r2/r3/r4）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在新建的 `paper-new-round5/` 代码分支中实现"长度自适应 genericity 惩罚"算法（Round 4 三段式 gate × 长度因子），通过 6 个新增单元测试 + α=0 sanity check + 离线钳制范围分布检查三重防护，跑完 r1/r2/r3/r4 共 16 个实验，验证"长 vs 短保护"哪个方向能让 forums 破 PrE-Text 基线。

**Architecture:**
1. `cp -r paper-new paper-new-round5` 物理隔离，避免污染 Round 4 已有结果
2. 在 `genericity.py` 新增 `compute_length_factors()` 函数；改 `compute_genericity_penalty` 与 `compute_genericity_penalties` 接受 length 参数
3. 在 `stage1_runner.py` 用 `len(text.split())` 计算 candidate_lengths（与已有 `private_lengths` 一致），透传给 genericity 函数
4. config 体系：1 base + 4 group + 16 leaf；alpha=0 时严格等价 Round 4 g1
5. 实验前必做：(a) 6 个 unit test 全 pass、(b) sanity check（α=0 forums 与 Round 4 g1 selected_texts 逐字符一致）、(c) 离线钳制范围分布检查

**Tech Stack:** Python 3 (keyword-only args, type hints), unittest 框架, YAML config inheritance, statistics.median, A6000 GPU on remote server (CUDA_VISIBLE_DEVICES=1)。

**前置依赖：**
- Spec：`paper-new/docs/2026-04-27-round5-dual-track-design.md` §3
- 已确认的 Round 4 代码位置：
  - `paper-new/paper_new_selector/genericity.py` 第 55-93 行 `compute_genericity_penalty`、第 96-121 行 `compute_genericity_penalties`
  - `paper-new/paper_new_selector/stage1_runner.py` 第 138 行 `private_lengths = [len(text.split()) for text in private_texts]`（candidate_lengths 应同样实现）
  - `paper-new/paper_new_selector/stage1_runner.py` 第 168-178 行调用 `compute_genericity_penalties` 的位置
- 远端服务器：`1u72c85740.zicp.fun:54360`，`k8smaster:k8s`，repo 在 `/mnt/public/caiqiyue_file/code_from_paper/`
- conda 环境：`pretext`，Python `/home/k8smaster/anaconda3/envs/pretext/bin/python`

---

## Task 1: 物理隔离 — copy paper-new 到 paper-new-round5

**Files:**
- Create: `paper-new-round5/` 整个目录树（从 `paper-new/` 拷贝）

- [ ] **Step 1: 检查目标目录不存在**

```bash
ls /Users/apple/Desktop/code_from_paper/paper-new-round5/ 2>&1 | head -1
```

Expected: `ls: ...: No such file or directory`（如果存在，先删 `rm -rf paper-new-round5/` 或备份）

- [ ] **Step 2: 拷贝整个目录**

```bash
cd /Users/apple/Desktop/code_from_paper
cp -r paper-new paper-new-round5
```

- [ ] **Step 3: 清理 outputs（不需要带过去，重新跑）**

```bash
rm -rf /Users/apple/Desktop/code_from_paper/paper-new-round5/outputs
```

- [ ] **Step 4: 验证关键文件齐全**

```bash
test -f /Users/apple/Desktop/code_from_paper/paper-new-round5/paper_new_selector/genericity.py && echo "genericity OK"
test -f /Users/apple/Desktop/code_from_paper/paper-new-round5/paper_new_selector/stage1_runner.py && echo "stage1_runner OK"
test -f /Users/apple/Desktop/code_from_paper/paper-new-round5/tests/test_support.py && echo "test_support OK"
test -f /Users/apple/Desktop/code_from_paper/paper-new-round5/tests/test_stage1_runner.py && echo "test_stage1_runner OK"
test -d /Users/apple/Desktop/code_from_paper/paper-new-round5/configs/experiments/single_node_tuning_round4 && echo "round4 configs OK"
```

Expected: 五行 `OK`。

- [ ] **Step 5: 跑一遍 paper-new-round5 现有 tests，确认 baseline**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
python -m unittest discover tests 2>&1 | tail -10
```

Expected: `OK` 或 `Ran X tests in Y.Zs` 后接 `OK`，无 FAIL/ERROR。

如果有 ERROR/FAIL，**停下来排查**——说明 copy 过程中丢失了文件或 import 路径问题。

- [ ] **Step 6: Commit**

```bash
cd /Users/apple/Desktop/code_from_paper
git add paper-new-round5
git commit -m "feat(round5): bootstrap paper-new-round5 from paper-new (copy for length-adaptive)"
```

注：如果 paper-new-round5 太大触发 commit hook 限制，分步 add：先 add `paper_new_selector/`、`tests/`、`configs/`、`thesis_platform/`、其他根文件，跳过 `__pycache__`。

---

## Task 2: TDD — 写 test 1 (alpha=0 时 length_factor=1.0)

**Files:**
- Create: `paper-new-round5/tests/test_length_modulation.py`

- [ ] **Step 1: 写第一个测试**

```python
import math
import unittest

from paper_new_selector.genericity import compute_length_factors


class LengthFactorTests(unittest.TestCase):
    def test_length_factor_neutral_when_alpha_zero(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=0.0,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        self.assertEqual(len(factors), 4)
        for factor in factors:
            self.assertAlmostEqual(factor, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试，确认 FAIL（函数还不存在）**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
python -m unittest tests.test_length_modulation -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'compute_length_factors'` 或 `AttributeError`。

- [ ] **Step 3: 实现最小可用 `compute_length_factors`**

修改 `paper-new-round5/paper_new_selector/genericity.py`，在文件末尾追加：

```python
def compute_length_factors(
    *,
    lengths: list[int],
    alpha: float,
    l_ref_strategy: str = "batch_median",
    factor_min: float = 0.2,
    factor_max: float = 5.0,
) -> list[float]:
    """Compute per-candidate length modulation factors for genericity penalty.

    factor(c) = clip( (L_ref / max(L_c, 1)) ^ alpha, factor_min, factor_max )

    L_ref strategy: 'batch_median' uses statistics.median(lengths).
    When alpha == 0, all factors are exactly 1.0 (constant short-circuit).
    """
    if not lengths:
        return []
    if alpha == 0.0:
        return [1.0] * len(lengths)
    if l_ref_strategy != "batch_median":
        raise ValueError(f"Unsupported l_ref_strategy: {l_ref_strategy}")

    import statistics
    l_ref = float(statistics.median(lengths))
    factors: list[float] = []
    for length in lengths:
        l_c = max(int(length), 1)
        ratio = l_ref / l_c
        raw = ratio ** alpha
        clipped = max(factor_min, min(factor_max, raw))
        factors.append(float(clipped))
    return factors
```

注意：keyword-only（`*`）保持与项目现有风格一致（参考 `compute_genericity_penalty`）。

- [ ] **Step 4: 跑测试，确认 PASS**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
python -m unittest tests.test_length_modulation.LengthFactorTests.test_length_factor_neutral_when_alpha_zero -v 2>&1 | tail -5
```

Expected: `OK`

但注意——上面的测试是用 keyword-only 参数调用 `compute_length_factors(lengths=..., alpha=...)`。Step 1 的 test 也得改成 keyword 形式。如果 Step 1 用了位置参数会失败，回去把 test 1 改成 `compute_length_factors(lengths=[5, 10, 20, 50], alpha=0.0, ...)`。

- [ ] **Step 5: Commit**

```bash
git add paper-new-round5/tests/test_length_modulation.py paper-new-round5/paper_new_selector/genericity.py
git commit -m "feat(round5): add compute_length_factors with alpha=0 short-circuit (test 1)"
```

---

## Task 3: TDD — test 2 (alpha>0 长候选 factor 更小)

**Files:**
- Modify: `paper-new-round5/tests/test_length_modulation.py`

- [ ] **Step 1: 在 test_length_modulation.py 的 LengthFactorTests 类中新增**

```python
    def test_length_factor_protects_longer_when_alpha_positive(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=0.3,
            l_ref_strategy="batch_median",
            factor_min=0.01,
            factor_max=100.0,
        )
        # batch median of [5, 10, 20, 50] = (10 + 20) / 2 = 15
        # factor(5)  = (15/5) ^ 0.3  = 3.0 ^ 0.3   ≈ 1.3904
        # factor(10) = (15/10) ^ 0.3 = 1.5 ^ 0.3   ≈ 1.1292
        # factor(20) = (15/20) ^ 0.3 = 0.75 ^ 0.3  ≈ 0.9163
        # factor(50) = (15/50) ^ 0.3 = 0.3 ^ 0.3   ≈ 0.6968
        # Expectation: longer candidate → smaller factor → LESS penalty
        self.assertGreater(factors[0], factors[1])
        self.assertGreater(factors[1], factors[2])
        self.assertGreater(factors[2], factors[3])
        self.assertAlmostEqual(factors[0], 3.0 ** 0.3, places=6)
        self.assertAlmostEqual(factors[3], 0.3 ** 0.3, places=6)
```

注意：把 `factor_min` 调成 0.01、`factor_max` 调成 100.0 是为了在测试中**关掉钳制**，单独验证 raw 公式正确性。钳制行为有专门的 test 4 验证。

- [ ] **Step 2: 跑测试**

```bash
python -m unittest tests.test_length_modulation.LengthFactorTests.test_length_factor_protects_longer_when_alpha_positive -v 2>&1 | tail -5
```

Expected: `OK`

如果 FAIL，先打印 `factors` 确认数学是否对：长候选的 factor 应当 < 短候选。如果方向反了，可能是 `(L_c / L_ref) ^ alpha` 而不是 `(L_ref / L_c) ^ alpha` 写错了 — 检查 `genericity.py` 实现。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/tests/test_length_modulation.py
git commit -m "test(round5): add length factor direction test for alpha>0"
```

---

## Task 4: TDD — test 3 (alpha<0 短候选 factor 更小)

- [ ] **Step 1: 新增测试**

在 `LengthFactorTests` 中追加：

```python
    def test_length_factor_protects_shorter_when_alpha_negative(self):
        factors = compute_length_factors(
            lengths=[5, 10, 20, 50],
            alpha=-0.3,
            l_ref_strategy="batch_median",
            factor_min=0.01,
            factor_max=100.0,
        )
        # batch median = 15
        # alpha=-0.3 means: factor = (L_ref/L_c)^(-0.3) = (L_c/L_ref)^0.3
        # factor(5)  = (5/15)^0.3  ≈ 0.7192
        # factor(50) = (50/15)^0.3 ≈ 1.4347
        # Expectation: shorter candidate → smaller factor → LESS penalty
        self.assertLess(factors[0], factors[1])
        self.assertLess(factors[1], factors[2])
        self.assertLess(factors[2], factors[3])
        self.assertAlmostEqual(factors[0], (5/15) ** 0.3, places=6)
```

- [ ] **Step 2: 跑测试**

```bash
python -m unittest tests.test_length_modulation.LengthFactorTests.test_length_factor_protects_shorter_when_alpha_negative -v 2>&1 | tail -5
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/tests/test_length_modulation.py
git commit -m "test(round5): add length factor direction test for alpha<0"
```

---

## Task 5: TDD — test 4 (极端长度被钳到 factor_min/max)

- [ ] **Step 1: 新增测试**

```python
    def test_length_factor_clipped_to_min_max(self):
        # 极短 + alpha=0.6: factor = (L_ref/1)^0.6 might explode → must clip to factor_max
        factors_short = compute_length_factors(
            lengths=[1, 100, 100],  # L_c=1 极短，median=100
            alpha=0.6,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        # raw factor for L_c=1: (100/1)^0.6 = 100^0.6 ≈ 15.85, clipped to 5.0
        self.assertAlmostEqual(factors_short[0], 5.0, places=9)

        # 极长 + alpha=0.6: factor = (L_ref/L_c)^0.6 might collapse → clip to factor_min
        factors_long = compute_length_factors(
            lengths=[10, 10, 1000],  # L_c=1000 极长，median=10
            alpha=0.6,
            l_ref_strategy="batch_median",
            factor_min=0.2,
            factor_max=5.0,
        )
        # raw factor for L_c=1000: (10/1000)^0.6 = 0.01^0.6 ≈ 0.0631, clipped to 0.2
        self.assertAlmostEqual(factors_long[2], 0.2, places=9)
```

- [ ] **Step 2: 跑测试**

```bash
python -m unittest tests.test_length_modulation.LengthFactorTests.test_length_factor_clipped_to_min_max -v 2>&1 | tail -5
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/tests/test_length_modulation.py
git commit -m "test(round5): add length factor clamp boundary test"
```

---

## Task 6: 修改 `compute_genericity_penalty` 接受 length 参数

**Files:**
- Modify: `paper-new-round5/paper_new_selector/genericity.py` 第 55-93 行

- [ ] **Step 1: 修改函数签名 + 内部逻辑**

把 `compute_genericity_penalty` 改成（注意 keyword-only 风格不变，新增 5 个 length 相关参数都默认值，旧调用代码完全兼容）：

```python
def compute_genericity_penalty(
    *,
    candidate_vector: list[float],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
    # ↓↓↓ Round 5 新增 ↓↓↓
    candidate_length: int | None = None,
    l_ref: float | None = None,
    length_modulation_enabled: bool = False,
    length_alpha: float = 0.0,
    length_factor_min: float = 0.2,
    length_factor_max: float = 5.0,
) -> float:
    """Estimate how close a candidate stays to the public initialization distribution."""

    if not reference_vectors:
        return 0.0
    top_scores = sorted(
        (_cosine(candidate_vector, reference) for reference in reference_vectors),
        reverse=True,
    )[: max(1, reference_top_k)]
    weights = _resolve_reference_rank_weights(
        count=len(top_scores),
        reference_rank_weights=reference_rank_weights,
    )
    denominator = float(sum(weights))
    if denominator <= 0.0:
        return 0.0
    weighted_mean = sum(score * weight for score, weight in zip(top_scores, weights)) / denominator
    raw_score = max(0.0, min(1.0, float(weighted_mean)))
    if not apply_gate:
        gated = raw_score
    else:
        gate_scale = apply_genericity_gate(
            score=raw_score,
            gate_low=gate_low,
            gate_high=gate_high,
            low_scale=low_scale,
            mid_scale=mid_scale,
        )
        gated = raw_score * gate_scale

    if length_modulation_enabled and candidate_length is not None and l_ref is not None and length_alpha != 0.0:
        l_c = max(int(candidate_length), 1)
        ratio = l_ref / l_c
        raw_factor = ratio ** length_alpha
        factor = max(length_factor_min, min(length_factor_max, raw_factor))
        gated = gated * factor

    return gated
```

注意：
- length 调制只在 `length_modulation_enabled=True AND candidate_length not None AND l_ref not None AND length_alpha != 0.0` 时生效
- alpha=0.0 时短路，避免不必要的 pow
- 参数复合默认 (`enabled=False`) 保证 Round 4 已有 4 个测试（test_genericity_penalty_*）继续 pass，无需改动

- [ ] **Step 2: 跑 Round 4 已有的全部 genericity 测试，确认无回归**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
python -m unittest tests.test_support -v 2>&1 | tail -15
```

Expected: 全部 PASS（包括 `test_genericity_penalty_is_high_for_public_template_like_candidates`、`test_genericity_penalty_supports_rank_weighted_reference_mean`、`test_genericity_gate_uses_low_mid_high_scales`、`test_genericity_penalty_applies_gate_to_raw_score` 等）。

如果有 FAIL，回去检查是否打破了 keyword-only 调用 — Round 4 测试用的 keyword arg。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/paper_new_selector/genericity.py
git commit -m "feat(round5): extend compute_genericity_penalty with length modulation params"
```

---

## Task 7: 修改 `compute_genericity_penalties` 计算 L_ref 并透传

**Files:**
- Modify: `paper-new-round5/paper_new_selector/genericity.py` 第 96-121 行

- [ ] **Step 1: 修改批处理函数**

替换原 `compute_genericity_penalties` 为：

```python
def compute_genericity_penalties(
    *,
    candidate_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
    # ↓↓↓ Round 5 新增 ↓↓↓
    candidate_lengths: list[int] | None = None,
    length_modulation_enabled: bool = False,
    length_alpha: float = 0.0,
    length_factor_min: float = 0.2,
    length_factor_max: float = 5.0,
) -> list[float]:
    # 仅当启用且非零 alpha 且有 lengths 时才计算 l_ref
    l_ref: float | None = None
    if length_modulation_enabled and candidate_lengths and length_alpha != 0.0:
        if len(candidate_lengths) != len(candidate_vectors):
            raise ValueError(
                f"candidate_lengths length ({len(candidate_lengths)}) "
                f"does not match candidate_vectors length ({len(candidate_vectors)})"
            )
        import statistics
        l_ref = float(statistics.median(candidate_lengths))

    lengths_iter = candidate_lengths if candidate_lengths is not None else [None] * len(candidate_vectors)
    return [
        compute_genericity_penalty(
            candidate_vector=candidate_vector,
            reference_vectors=reference_vectors,
            reference_top_k=reference_top_k,
            reference_rank_weights=reference_rank_weights,
            apply_gate=apply_gate,
            gate_low=gate_low,
            gate_high=gate_high,
            low_scale=low_scale,
            mid_scale=mid_scale,
            candidate_length=length,
            l_ref=l_ref,
            length_modulation_enabled=length_modulation_enabled,
            length_alpha=length_alpha,
            length_factor_min=length_factor_min,
            length_factor_max=length_factor_max,
        )
        for candidate_vector, length in zip(candidate_vectors, lengths_iter)
    ]
```

关键点：
- `l_ref` 只在调制启用且 alpha 非零时计算（避免不必要的 statistics 调用）
- 长度不匹配时显式 raise ValueError，防止 silent bug
- 旧调用（不传 length_*）走原路径，零行为变化

- [ ] **Step 2: 再跑一次 Round 4 测试，确保仍 pass**

```bash
python -m unittest tests.test_support -v 2>&1 | tail -10
```

Expected: 全部 PASS。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/paper_new_selector/genericity.py
git commit -m "feat(round5): extend compute_genericity_penalties with batch median L_ref"
```

---

## Task 8: TDD — test 5 (disabled 时严格等价 Round 4)

**Files:**
- Modify: `paper-new-round5/tests/test_length_modulation.py`

- [ ] **Step 1: 新增关键回归测试**

```python
    def test_genericity_with_length_disabled_matches_round4(self):
        """When length_modulation_enabled=False, output must EXACTLY match Round 4 behavior."""
        from paper_new_selector.genericity import compute_genericity_penalties

        candidate_vectors = [
            [1.0, 0.0],
            [0.7, 0.714142842854285],
            [0.0, 1.0],
        ]
        reference_vectors = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]]
        common_kwargs = dict(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
            apply_gate=True,
            gate_low=0.78,
            gate_high=0.90,
            low_scale=0.10,
            mid_scale=0.45,
        )

        # Round 4 风格调用（不传 length 参数）
        baseline = compute_genericity_penalties(**common_kwargs)

        # Round 5 disabled（显式传，但 enabled=False）
        with_disabled = compute_genericity_penalties(
            **common_kwargs,
            candidate_lengths=[5, 10, 20],
            length_modulation_enabled=False,
            length_alpha=0.6,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )

        # Round 5 enabled but alpha=0
        with_alpha_zero = compute_genericity_penalties(
            **common_kwargs,
            candidate_lengths=[5, 10, 20],
            length_modulation_enabled=True,
            length_alpha=0.0,
            length_factor_min=0.2,
            length_factor_max=5.0,
        )

        for i in range(3):
            self.assertAlmostEqual(baseline[i], with_disabled[i], places=12)
            self.assertAlmostEqual(baseline[i], with_alpha_zero[i], places=12)
```

- [ ] **Step 2: 跑测试**

```bash
python -m unittest tests.test_length_modulation.LengthFactorTests.test_genericity_with_length_disabled_matches_round4 -v 2>&1 | tail -5
```

Expected: `OK`

如果 FAIL — 说明在 Task 6/7 中引入了非零路径下的副作用。检查 `compute_genericity_penalty` 是否在 `length_modulation_enabled=False` 或 `length_alpha=0.0` 时**完全跳过** length 计算分支。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/tests/test_length_modulation.py
git commit -m "test(round5): add critical regression test for disabled-equals-round4"
```

---

## Task 9: 修改 stage1_runner.py 透传 length 配置

**Files:**
- Modify: `paper-new-round5/paper_new_selector/stage1_runner.py` 第 168-178 行附近

- [ ] **Step 1: 在调用 compute_genericity_penalties 前增加 candidate_lengths 计算**

在 `paper-new-round5/paper_new_selector/stage1_runner.py` 中找到第 168 行 `genericity_penalty = compute_genericity_penalties(`，在它**之前**插入 candidate_lengths 计算（位置在 `private_lengths = [len(text.split()) for text in private_texts]` 第 138 行附近的同一段函数体内，但要在 candidate_texts 已经被 cleaned/truncated 之后，即 `candidate_texts = candidate_texts[:candidate_count]` 第 131 行之后）。

具体修改如下：

**原 168-178 行**：
```python
        genericity_penalty = compute_genericity_penalties(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=int(selector_cfg["reference_top_k"]),
            reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
            apply_gate=True,
            gate_low=float(selector_cfg.get("genericity_gate_low", 0.0)),
            gate_high=float(selector_cfg.get("genericity_gate_high", 1.0)),
            low_scale=float(selector_cfg.get("genericity_gate_low_scale", 1.0)),
            mid_scale=float(selector_cfg.get("genericity_gate_mid_scale", 1.0)),
        )
```

**改为**：
```python
        candidate_lengths = [len(text.split()) for text in candidate_texts]
        genericity_penalty = compute_genericity_penalties(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=int(selector_cfg["reference_top_k"]),
            reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
            apply_gate=True,
            gate_low=float(selector_cfg.get("genericity_gate_low", 0.0)),
            gate_high=float(selector_cfg.get("genericity_gate_high", 1.0)),
            low_scale=float(selector_cfg.get("genericity_gate_low_scale", 1.0)),
            mid_scale=float(selector_cfg.get("genericity_gate_mid_scale", 1.0)),
            candidate_lengths=candidate_lengths,
            length_modulation_enabled=bool(selector_cfg.get("length_modulation_enabled", False)),
            length_alpha=float(selector_cfg.get("length_alpha", 0.0)),
            length_factor_min=float(selector_cfg.get("length_factor_min", 0.2)),
            length_factor_max=float(selector_cfg.get("length_factor_max", 5.0)),
        )
```

注意 `.get(..., 默认值)` 模式 — 旧 config 不传这些字段也能跑（默认 disabled）。

- [ ] **Step 2: 跑现有 stage1_runner 测试，确保无回归**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
python -m unittest tests.test_stage1_runner -v 2>&1 | tail -10
```

Expected: 全部 PASS（包括 `test_stage1_runner_passes_genericity_gate_config_to_genericity`）。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/paper_new_selector/stage1_runner.py
git commit -m "feat(round5): pass length modulation config from stage1_runner to genericity"
```

---

## Task 10: TDD — test 6 (stage1_runner 透传 length 配置)

**Files:**
- Modify: `paper-new-round5/tests/test_stage1_runner.py`

- [ ] **Step 1: 在文件末尾的 Stage1RunnerTests 类中新增**

参考已有 `test_stage1_runner_passes_genericity_gate_config_to_genericity`（第 322 行）的模式，添加 length 透传测试：

```python
    def test_stage1_runner_passes_length_modulation_config_to_genericity(self):
        fake_backend = _FakeTextBackend()
        fake_embedder = _FakeEmbedder()
        config = {
            "pipeline": {"stage1_mode": "selector_seed_search"},
            "generator": {
                "initial_prompt": "prompt",
                "candidate_count": 2,
                "max_rounds": 1,
                "exemplars_per_prompt": 1,
            },
            "meta": {"seed": 42},
            "selector": {
                "private_knn_k": 1,
                "density_lambda": 0.0,
                "novelty_lambda": 0.0,
                "length_lambda": 0.0,
                "length_floor": 1,
                "length_ceiling": 100,
                "rank_weights": [1.0],
                "top_q": 1,
                "reference_top_k": 6,
                "reference_rank_weights": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
                "genericity_gate_low": 0.78,
                "genericity_gate_high": 0.90,
                "genericity_gate_low_scale": 0.10,
                "genericity_gate_mid_scale": 0.45,
                # ↓↓↓ Round 5 新增字段 ↓↓↓
                "length_modulation_enabled": True,
                "length_alpha": 0.3,
                "length_factor_min": 0.2,
                "length_factor_max": 5.0,
                "lambda_generic": 0.2,
                "lambda_redundancy": 0.3,
                "seed_top_k": 1,
                "hard_negative_top_k": 1,
            },
            "privacy": {"enabled": False, "delta": 1e-5},
            "stage1": {"sigma": 0.0, "delta": 1e-5},
        }
        sample_bundle = {
            "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
            "eval_samples": [_FakeSample("eval alpha")],
            "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
        }
        decision = SimpleNamespace(
            seed_indices=[0],
            hard_negative_indices=[1],
            selected_scores={0: 0.7, 1: 0.3},
            seed_score_breakdown=[],
            hard_negative_score_breakdown=[],
        )
        with patch(
            "paper_new_selector.stage1_runner._load_config",
            return_value=config,
        ), patch(
            "paper_new_selector.stage1_runner._load_sample_bundle",
            return_value=sample_bundle,
        ), patch(
            "paper_new_selector.stage1_runner._build_generator_handle",
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
            ),
        ), patch(
            "paper_new_selector.stage1_runner._build_embedder",
            return_value=fake_embedder,
        ), patch(
            "paper_new_selector.stage1_runner.build_private_importance_weights",
            return_value=[1.0, 1.0],
        ), patch(
            "paper_new_selector.stage1_runner.compute_private_support",
            return_value=[0.9, 0.2],
        ), patch(
            "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
            side_effect=lambda scores, **_: scores,
        ), patch(
            "paper_new_selector.stage1_runner.compute_genericity_penalties",
            return_value=[0.1, 0.3],
        ) as genericity_mock, patch(
            "paper_new_selector.stage1_runner.greedy_select_candidates",
            return_value=decision,
        ), patch(
            "paper_new_selector.stage1_runner.build_boundary_state",
            return_value={"negative_pattern_stats": {"count": 1}},
        ):
            run_stage1("dummy.yaml", validate_only=False)

        kwargs = genericity_mock.call_args.kwargs
        self.assertTrue(kwargs["length_modulation_enabled"])
        self.assertEqual(kwargs["length_alpha"], 0.3)
        self.assertEqual(kwargs["length_factor_min"], 0.2)
        self.assertEqual(kwargs["length_factor_max"], 5.0)
        # candidate_lengths is computed from candidate_texts via len(text.split())
        # _FakeGenerator emits "candidate alpha text" (3 words) and "candidate beta text" (3 words)
        self.assertEqual(kwargs["candidate_lengths"], [3, 3])
```

- [ ] **Step 2: 跑测试**

```bash
python -m unittest tests.test_stage1_runner.Stage1RunnerTests.test_stage1_runner_passes_length_modulation_config_to_genericity -v 2>&1 | tail -10
```

Expected: `OK`

- [ ] **Step 3: 再跑一次全部 tests，确认无回归**

```bash
python -m unittest discover tests 2>&1 | tail -5
```

Expected: `Ran X tests in Y.Zs` 后 `OK`，无 FAIL/ERROR。

- [ ] **Step 4: Commit**

```bash
git add paper-new-round5/tests/test_stage1_runner.py
git commit -m "test(round5): verify stage1_runner passes length modulation config to genericity"
```

---

## Task 11: 写 Round 5 base config 和 4 个组配置

**Files:**
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_base_selector_tuning_round5.yaml`
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_r1_protect_long_moderate.yaml`
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_r2_protect_short_moderate.yaml`
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_r3_protect_long_strong.yaml`
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_r4_protect_short_strong.yaml`

- [ ] **Step 1: 新建目录**

```bash
mkdir -p /Users/apple/Desktop/code_from_paper/paper-new-round5/configs/experiments/single_node_tuning_round5
```

- [ ] **Step 2: 写 base config**

文件 `_base_selector_tuning_round5.yaml`：

```yaml
inherits:
  - ../single_node_tuning_round4/_base_selector_tuning_round4.yaml

meta:
  stage: single_node_tuning_round5

selector:
  # Round 5: 长度自适应惩罚（默认 disabled，由各 r 组独立打开）
  length_modulation_enabled: false
  length_alpha: 0.0
  length_factor_min: 0.2
  length_factor_max: 5.0
```

- [ ] **Step 3: 写 r1（长保护 适度）**

文件 `_r1_protect_long_moderate.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round5.yaml

selector:
  length_modulation_enabled: true
  length_alpha: 0.3
```

- [ ] **Step 4: 写 r2（短保护 适度）**

文件 `_r2_protect_short_moderate.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round5.yaml

selector:
  length_modulation_enabled: true
  length_alpha: -0.3
```

- [ ] **Step 5: 写 r3（长保护 强）**

文件 `_r3_protect_long_strong.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round5.yaml

selector:
  length_modulation_enabled: true
  length_alpha: 0.6
```

- [ ] **Step 6: 写 r4（短保护 强）**

文件 `_r4_protect_short_strong.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round5.yaml

selector:
  length_modulation_enabled: true
  length_alpha: -0.6
```

- [ ] **Step 7: 验证 5 个 yaml 都可解析**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round5
for f in configs/experiments/single_node_tuning_round5/*.yaml; do
  python -c "import yaml; yaml.safe_load(open('$f'))" && echo "OK: $f" || echo "FAIL: $f"
done
```

Expected: 5 行 `OK`。

- [ ] **Step 8: Commit**

```bash
git add paper-new-round5/configs/experiments/single_node_tuning_round5/_base_*.yaml \
        paper-new-round5/configs/experiments/single_node_tuning_round5/_r*.yaml
git commit -m "feat(round5): add round5 base + r1-r4 group configs (length modulation alpha=±0.3, ±0.6)"
```

---

## Task 12: 写 r0 sanity check 配置（α=0 forums）

**Files:**
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/_r0_sanity_alpha_zero.yaml`
- Create: `paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r0_forums.yaml`

- [ ] **Step 1: 写 r0 组配置（length_modulation_enabled=true 但 alpha=0）**

文件 `_r0_sanity_alpha_zero.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round5.yaml

selector:
  length_modulation_enabled: true
  length_alpha: 0.0
```

注意：这个组的存在意义是验证"开启长度调制 + alpha=0"完全等价于"关闭长度调制"。

- [ ] **Step 2: 写 r0 forums 叶子 config**

文件 `ns_tune5_r0_forums.yaml`：

```yaml
inherits:
  - ./_r0_sanity_alpha_zero.yaml

meta:
  experiment_id: ns_tune5_r0_forums

paths:
  output_root: paper-new/outputs/ns_tune5_r0_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

注意：`output_root` 用 `paper-new/outputs/...` 是因为远端服务器 paper-new-round5 跑实验时也会写入那个相对路径下的目录（paper-new-round5 内的相对路径）。

- [ ] **Step 3: Commit**

```bash
git add paper-new-round5/configs/experiments/single_node_tuning_round5/_r0_*.yaml \
        paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r0_forums.yaml
git commit -m "feat(round5): add r0 sanity check config (alpha=0 forums)"
```

---

## Task 13: 写 r1-r4 共 16 个叶子配置

**Files:**（在 `paper-new-round5/configs/experiments/single_node_tuning_round5/`）
- 16 个 leaf：`ns_tune5_r{1,2,3,4}_{jobs,congressional,forums,microblog}.yaml`

- [ ] **Step 1: 写 r1 的 4 个叶子**

文件 `ns_tune5_r1_jobs.yaml`：
```yaml
inherits:
  - ./_r1_protect_long_moderate.yaml

meta:
  experiment_id: ns_tune5_r1_jobs

paths:
  output_root: paper-new/outputs/ns_tune5_r1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

文件 `ns_tune5_r1_congressional.yaml`：
```yaml
inherits:
  - ./_r1_protect_long_moderate.yaml

meta:
  experiment_id: ns_tune5_r1_congressional

paths:
  output_root: paper-new/outputs/ns_tune5_r1_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

文件 `ns_tune5_r1_forums.yaml`：
```yaml
inherits:
  - ./_r1_protect_long_moderate.yaml

meta:
  experiment_id: ns_tune5_r1_forums

paths:
  output_root: paper-new/outputs/ns_tune5_r1_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

文件 `ns_tune5_r1_microblog.yaml`：
```yaml
inherits:
  - ./_r1_protect_long_moderate.yaml

meta:
  experiment_id: ns_tune5_r1_microblog

paths:
  output_root: paper-new/outputs/ns_tune5_r1_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 2: 写 r2 的 4 个叶子**

复用 Step 1 的 4 个模板，全部把 `_r1_protect_long_moderate` 换成 `_r2_protect_short_moderate`，把 `ns_tune5_r1_*` 换成 `ns_tune5_r2_*`，包括 `experiment_id` 和 `output_root`。

例如 `ns_tune5_r2_forums.yaml`：
```yaml
inherits:
  - ./_r2_protect_short_moderate.yaml

meta:
  experiment_id: ns_tune5_r2_forums

paths:
  output_root: paper-new/outputs/ns_tune5_r2_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

其他 3 个数据集（jobs/congressional/microblog）以此类推。

- [ ] **Step 3: 写 r3 的 4 个叶子**

同 Step 2 模式，inherits 改 `_r3_protect_long_strong`，前缀改 `r3`。

- [ ] **Step 4: 写 r4 的 4 个叶子**

同 Step 2 模式，inherits 改 `_r4_protect_short_strong`，前缀改 `r4`。

- [ ] **Step 5: 验证总数**

```bash
ls /Users/apple/Desktop/code_from_paper/paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r*.yaml | wc -l
```

Expected: `17`（16 个 r1-r4 + 1 个 r0_forums）

- [ ] **Step 6: Commit**

```bash
git add paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r1_*.yaml \
        paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r2_*.yaml \
        paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r3_*.yaml \
        paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r4_*.yaml
git commit -m "feat(round5): add r1-r4 leaf configs (16 experiments × 4 datasets)"
```

---

## Task 14: 写 run_round5_queue.py

**Files:**
- Create: `old_automation/run_round5_queue.py`

- [ ] **Step 1: 写 runner**

```python
#!/usr/bin/env python3
"""Sequential runner for Round 5 length-adaptive experiments (16 total: r1/r2/r3/r4 × 4 datasets) on A6000 GPU."""
import datetime
import os
import subprocess
import sys

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW_ROUND5 = REPO + "/paper-new-round5"
AUTOMATION = REPO + "/old_automation"
LOG_PATH = AUTOMATION + "/run_round5_queue.log"

EXPERIMENTS = [
    ("r1", "jobs"), ("r1", "congressional"), ("r1", "forums"), ("r1", "microblog"),
    ("r2", "jobs"), ("r2", "congressional"), ("r2", "forums"), ("r2", "microblog"),
    ("r3", "jobs"), ("r3", "congressional"), ("r3", "forums"), ("r3", "microblog"),
    ("r4", "jobs"), ("r4", "congressional"), ("r4", "forums"), ("r4", "microblog"),
]

ENV = {
    **os.environ,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1",
    "PYTHONUNBUFFERED": "1",
    "VLLM_HOST_IP": "127.0.0.1",
    "HOST_IP": "127.0.0.1",
}

PYTHON = "/home/k8smaster/anaconda3/envs/pretext/bin/python"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_experiment(group, dataset):
    exp_id = f"ns_tune5_{group}_{dataset}"
    config = f"configs/experiments/single_node_tuning_round5/{exp_id}.yaml"
    remote_log = f"{AUTOMATION}/NS-T5-{group.upper()}-{dataset.upper()}.remote.log"
    log(f"Starting {exp_id}")
    log(f"Config: {config}")
    with open(remote_log, "w") as out:
        result = subprocess.run(
            [PYTHON, "-m", "paper_new_selector.run_selector_single_node",
             "--config", config],
            cwd=PAPER_NEW_ROUND5,
            env=ENV,
            stdout=out,
            stderr=out,
        )
    if result.returncode == 0:
        log(f"SUCCESS: {exp_id}")
        return True
    else:
        log(f"FAILED (exit {result.returncode}): {exp_id} -- see {remote_log}")
        return False


def main():
    total = len(EXPERIMENTS)
    done = 0
    failed = 0
    log(f"=== Round 5 Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ===")
    exp_ids = [f"ns_tune5_{g}_{d}" for g, d in EXPERIMENTS]
    log(f"Queue: {exp_ids}")

    for i, (group, dataset) in enumerate(EXPERIMENTS, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(group, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 5 Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
```

注意 Round 4 runner 与 Round 5 runner 的差异：
- `cwd` 改为 `paper-new-round5/`
- `EXPERIMENTS` 改为 r1-r4 × 4 = 16 项
- config 路径 `single_node_tuning_round5/`
- exp_id 前缀 `ns_tune5_`
- log 前缀 `NS-T5-`

- [ ] **Step 2: 语法检查**

```bash
cd /Users/apple/Desktop/code_from_paper
python -c "import ast; ast.parse(open('old_automation/run_round5_queue.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 验证 EXPERIMENTS 数量**

```bash
python -c "
import ast
tree = ast.parse(open('old_automation/run_round5_queue.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(t.id == 'EXPERIMENTS' for t in node.targets if hasattr(t, 'id')):
        print(f'EXPERIMENTS count: {len(node.value.elts)}')
"
```

Expected: `EXPERIMENTS count: 16`

- [ ] **Step 4: Commit**

```bash
git add old_automation/run_round5_queue.py
git commit -m "feat(round5): add run_round5_queue.py for r1-r4 sequential execution"
```

---

## Task 15: 同步 paper-new-round5 + runner 到远端服务器

服务器连接：`1u72c85740.zicp.fun:54360`，`k8smaster:k8s`，repo 在 `/mnt/public/caiqiyue_file/code_from_paper/`。

- [ ] **Step 1: 上传 paper-new-round5 整个目录**

paper-new-round5 是个大目录（数 GB？），用 rsync 比 scp 高效。如果远端无 rsync，用 scp -r。

```bash
cd /Users/apple/Desktop/code_from_paper

# 选项 A: rsync（推荐）
sshpass -p 'k8s' rsync -avz -e "ssh -p 54360" \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='outputs' \
    paper-new-round5/ \
    k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/

# 选项 B: scp（如果 rsync 不可用）
# sshpass -p 'k8s' scp -P 54360 -r paper-new-round5 \
#     k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/
```

- [ ] **Step 2: 上传 runner**

```bash
sshpass -p 'k8s' scp -P 54360 \
    old_automation/run_round5_queue.py \
    k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/old_automation/
```

- [ ] **Step 3: 远端验证**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'ls /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/configs/experiments/single_node_tuning_round5/ns_tune5_r*.yaml | wc -l && \
     test -f /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round5_queue.py && echo "runner OK"'
```

Expected: 第一行 `17`（16 r1-r4 + 1 r0_forums），第二行 `runner OK`。

- [ ] **Step 4: 远端跑 unit tests，确保 paper-new-round5 在服务器上能跑**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'cd /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5 && /home/k8smaster/anaconda3/envs/pretext/bin/python -m unittest discover tests 2>&1 | tail -5'
```

Expected: `OK` 全部通过。

如果有 FAIL，说明本地 vs 远端 Python 环境差异 — 必须先解决再继续。

---

## Task 16: 跑 r0 sanity check（α=0 forums）

- [ ] **Step 1: 检查 Round 4 g1 forums 的对照基准是否存在**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'test -d /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_g1_forums && \
     ls /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_g1_forums/eval/stage2/llama7b_text_syn.json'
```

Expected: 文件存在；如果不存在，**先重跑一次 Round 4 g1 forums 作为基准**（约 5 分钟）。

- [ ] **Step 2: 跑 r0_forums sanity 实验**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'cd /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5 && \
     /home/k8smaster/anaconda3/envs/pretext/bin/python -m paper_new_selector.run_selector_single_node \
     --config configs/experiments/single_node_tuning_round5/ns_tune5_r0_forums.yaml \
     2>&1 | tee /mnt/public/caiqiyue_file/code_from_paper/old_automation/NS-T5-R0-FORUMS-SANITY.remote.log | tail -30'
```

注意 paper-new-round5 的 output_root 配置写的是 `paper-new/outputs/ns_tune5_r0_forums`，所以输出会落在 `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_r0_forums/` —— 实际路径以 cwd 为根，确认远端 cwd=paper-new-round5 时这是正确的。

- [ ] **Step 3: 多层一致性比对（spec §3.6 (a)(b)(c) 三项必须全过）**

**(a) Stage 1 selected_texts 文本一致性**：

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'diff /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_g1_forums/stage1/selected_texts.json \
          /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_r0_forums/stage1/selected_texts.json && \
     echo "STAGE1_SELECTED_OK"'
```

Expected: `STAGE1_SELECTED_OK`，无 diff 输出（逐字符一致）。

**(b) Stage 1 hard_negative_texts 文本一致性**：

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'diff /mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_g1_forums/stage1/hard_negative_texts.json \
          /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_r0_forums/stage1/hard_negative_texts.json && \
     echo "STAGE1_HARD_NEG_OK"'
```

Expected: `STAGE1_HARD_NEG_OK`。

注意：实际 stage1 输出的 json 文件名以现有代码为准。如果不叫 `selected_texts.json`，先在远端 `ls outputs/ns_tune4_g1_forums/stage1/` 看真实文件名再调整 diff 命令。

**(c) 下游评估 best_top1 数值差异 < 0.0001**：

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'python -c "
import json
g1 = json.load(open(\"/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_g1_forums/eval/downstream_eval_summary.json\"))
r0 = json.load(open(\"/mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_r0_forums/eval/downstream_eval_summary.json\"))
g1_top1 = g1.get(\"best\", {}).get(\"top1\", g1.get(\"best_top1\"))
r0_top1 = r0.get(\"best\", {}).get(\"top1\", r0.get(\"best_top1\"))
diff = abs(float(g1_top1) - float(r0_top1))
print(f\"g1_top1={g1_top1}, r0_top1={r0_top1}, diff={diff}\")
assert diff < 0.0001, f\"diff {diff} >= 0.0001 — sanity check FAILED\"
print(\"BEST_TOP1_OK\")
"'
```

Expected: 输出 `BEST_TOP1_OK`，且 diff 极小（理论应当为 0.0）。

- [ ] **Step 4: 三项全过才能继续**

如果 (a) 或 (b) 输出有 diff，**立即停下**——说明代码改动引入了数值漂移，必须排查 `genericity.py` 是否在 `length_modulation_enabled=true, alpha=0` 路径上仍然走了 length 计算分支。修复后回到 Task 8 重跑 test 5、再回到 Task 16 Step 2-3。

如果 (c) 数值差超过 0.0001，再排查 evaluator 路径。

- [ ] **Step 5: Commit sanity 结果（不上传 outputs，只是文档）**

在 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md` 末尾添加附录 B（已通过 sanity check 的标志）：

文件 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md`（如果还没建，从 paper-new/docs/2026-04-27-round5-dual-track-design.md 复制 §3 部分）。

```bash
cd /Users/apple/Desktop/code_from_paper
mkdir -p paper-new-round5/docs
cp paper-new/docs/2026-04-27-round5-dual-track-design.md paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md
```

在 paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md 末尾追加：

```markdown

---

## 附录 B：sanity check 通过记录（YYYY-MM-DD）

- (a) Stage 1 selected_texts 与 Round 4 g1 forums 逐字符一致：✅
- (b) Stage 1 hard_negative_texts 一致：✅
- (c) best_top1 数值差异：<填实际差> （< 0.0001 通过）

确认 length_modulation 在 alpha=0 时严格等价 Round 4 g1，可以进入 r1-r4 正式实验。
```

```bash
git add paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md
git commit -m "docs(round5): record sanity check pass for r0 (alpha=0 equals round4 g1)"
```

---

## Task 17: 离线钳制范围分布检查（spec §3.4.1）

**Files:**
- Create: `paper-new-round5/scripts/check_length_factor_distribution.py`（一次性脚本）

- [ ] **Step 1: 写检查脚本**

```python
#!/usr/bin/env python3
"""Offline check: how often does (L_ref/L_c)^alpha hit factor_min/factor_max boundaries?

Run after generating candidate_texts in stage1 for forums (largest length variance).
Reads stage1 candidate_texts dump and computes saturation rate for alpha=±0.6.
"""
import json
import statistics
import sys
from pathlib import Path


def load_candidate_lengths(stage1_dir: Path) -> list[int]:
    # Try common locations for candidate texts
    for filename in ("candidate_texts.json", "candidates.json", "selected_texts.json"):
        candidate_file = stage1_dir / filename
        if candidate_file.exists():
            data = json.load(open(candidate_file))
            if isinstance(data, list):
                return [len(t.split()) for t in data if isinstance(t, str)]
    raise FileNotFoundError(f"No candidate text file found in {stage1_dir}")


def check_saturation(lengths: list[int], alpha: float, factor_min: float, factor_max: float) -> dict:
    l_ref = float(statistics.median(lengths))
    factors_raw = []
    saturated_low = 0
    saturated_high = 0
    for l_c in lengths:
        l_c = max(int(l_c), 1)
        ratio = l_ref / l_c
        raw = ratio ** alpha
        factors_raw.append(raw)
        if raw <= factor_min:
            saturated_low += 1
        elif raw >= factor_max:
            saturated_high += 1
    n = len(lengths)
    return {
        "alpha": alpha,
        "l_ref": l_ref,
        "n": n,
        "raw_min": min(factors_raw),
        "raw_max": max(factors_raw),
        "raw_median": statistics.median(factors_raw),
        "saturated_low_pct": 100.0 * saturated_low / n,
        "saturated_high_pct": 100.0 * saturated_high / n,
        "in_band_pct": 100.0 * (n - saturated_low - saturated_high) / n,
    }


def main():
    stage1_dir = Path(sys.argv[1])  # e.g., outputs/ns_tune5_r0_forums/stage1/
    lengths = load_candidate_lengths(stage1_dir)
    print(f"Loaded {len(lengths)} candidate lengths from {stage1_dir}")
    print(f"Length stats: min={min(lengths)}, max={max(lengths)}, median={statistics.median(lengths)}, mean={statistics.mean(lengths):.1f}")
    print()
    for alpha in [0.3, -0.3, 0.6, -0.6]:
        result = check_saturation(lengths, alpha, factor_min=0.2, factor_max=5.0)
        in_band = result["in_band_pct"]
        flag = "OK" if in_band >= 80.0 else "WARN"
        print(f"alpha={alpha:+.1f}: in_band={in_band:.1f}% (saturated_low={result['saturated_low_pct']:.1f}%, saturated_high={result['saturated_high_pct']:.1f}%), raw_range=[{result['raw_min']:.3f}, {result['raw_max']:.3f}] [{flag}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 上传脚本到远端**

```bash
sshpass -p 'k8s' scp -P 54360 \
    paper-new-round5/scripts/check_length_factor_distribution.py \
    k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/scripts/check_length_factor_distribution.py
```

如果 `scripts/` 目录不存在，先 `mkdir`：

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'mkdir -p /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/scripts'
```

- [ ] **Step 3: 在远端跑分布检查（用 r0_forums 的 stage1 输出）**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    '/home/k8smaster/anaconda3/envs/pretext/bin/python \
     /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/scripts/check_length_factor_distribution.py \
     /mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_r0_forums/stage1/'
```

Expected: 输出 4 行 alpha=±0.3, ±0.6 的饱和率。**至少 α=±0.6 时 in_band >= 80% 才算 OK**。

- [ ] **Step 4: 决策 — 是否需要调整 factor_min/max 或 α**

如果 α=±0.6 时 in_band < 80%（即 >20% 的候选词被钳到边界）：
- **方案 A**：把 factor_min 调到 0.1、factor_max 调到 10.0，重新跑 Step 3 检查
- **方案 B**：把 r3/r4 的 α 从 ±0.6 改为 ±0.5 或 ±0.4（修改 `_r3_protect_long_strong.yaml` 和 `_r4_protect_short_strong.yaml`）
- **方案 C**：accept saturation but document it explicitly（如果分布检查显示 only ~70% in band，但 r3/r4 仍能在 r1/r2 之上提供方向信号，可能可以接受）

把决策记录到 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md` 的"附录 A：钳制范围离线检查"。

- [ ] **Step 5: Commit 检查脚本和决策记录**

```bash
cd /Users/apple/Desktop/code_from_paper
git add paper-new-round5/scripts/check_length_factor_distribution.py \
        paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md
git commit -m "feat(round5): add offline factor distribution check script + decision record"
```

如果做了方案 A/B 调整，把对应 yaml 也加进 commit。

---

## Task 18: 跑 r1-r4 共 16 个正式实验

- [ ] **Step 1: 在远端 tmux session 启动队列**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tmux new-session -d -s round5 "/home/k8smaster/anaconda3/envs/pretext/bin/python /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round5_queue.py"'
```

- [ ] **Step 2: 验证启动**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tmux list-sessions && tail -10 /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round5_queue.log'
```

Expected: `round5` session 存在；log 显示 `=== Round 5 Queue Start: 16 experiments ...`。

- [ ] **Step 3: 周期性查看进度（每 15 分钟一次，预计 ~80 分钟）**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tail -30 /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round5_queue.log'
```

观察 `[N/16]` 进度，关注任何 `FAILED`。

- [ ] **Step 4: 检测全部完成**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'grep "Round 5 Done" /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round5_queue.log'
```

Expected: `=== Round 5 Done: 16 success, 0 failed out of 16 ===`

如果失败 > 0，查 `NS-T5-*-*.remote.log` 排查个例。

---

## Task 19: 拉取结果并写入 paper-new-round5 文档

**Files:**
- Modify: `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md`（追加结果章节）
- Create: `paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md`

- [ ] **Step 1: 拉取 16 个 downstream_eval_summary.json**

```bash
mkdir -p /tmp/round5_results
for r in r1 r2 r3 r4; do
  for d in jobs congressional forums microblog; do
    sshpass -p 'k8s' scp -P 54360 \
      "k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/paper-new-round5/paper-new/outputs/ns_tune5_${r}_${d}/eval/downstream_eval_summary.json" \
      "/tmp/round5_results/ns_tune5_${r}_${d}.json" 2>&1 | grep -v "100%" || true
  done
done
ls /tmp/round5_results/ | wc -l
```

Expected: `16`

- [ ] **Step 2: 解析 best_top1/3/5/10**

```bash
for f in /tmp/round5_results/*.json; do
  exp=$(basename "$f" .json)
  python -c "
import json
d = json.load(open('$f'))
b = d.get('best', {})
print(f'$exp\t{b.get(\"top1\", b.get(\"best_top1\", \"?\"))}\t{b.get(\"top3\", b.get(\"best_top3\", \"?\"))}\t{b.get(\"top5\", b.get(\"best_top5\", \"?\"))}\t{b.get(\"top10\", b.get(\"best_top10\", \"?\"))}'
"
done
```

把 16 行输出整理成下面的表格格式。

- [ ] **Step 3: 写 results 文档**

文件 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md`：

```markdown
# Round 5 方向 2a 实验结果：长度自适应惩罚（r1/r2/r3/r4）

**日期**：YYYY-MM-DD
**配置**：见 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md`
**前置 sanity**：见 §附录 B（α=0 与 Round 4 g1 forums 逐字符一致 ✅）
**钳制分布检查**：见 §附录 A（α=±0.6 in_band ≥ 80% / 或调整记录）

## 1. 完整结果表

| 实验 ID | 组别 | α | 数据集 | best_top1 | best_top3 | best_top5 | best_top10 |
|---------|------|---|--------|-----------|-----------|-----------|------------|
| ns_tune5_r1_jobs | r1 | +0.3 | jobs | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> |
| ns_tune5_r1_congressional | r1 | +0.3 | congressional | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> |
| ns_tune5_r1_forums | r1 | +0.3 | forums | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> |
| ns_tune5_r1_microblog | r1 | +0.3 | microblog | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> | <Step 2 数值> |
| ns_tune5_r2_jobs | r2 | -0.3 | jobs | <填> | <填> | <填> | <填> |
| ns_tune5_r2_congressional | r2 | -0.3 | congressional | <填> | <填> | <填> | <填> |
| ns_tune5_r2_forums | r2 | -0.3 | forums | <填> | <填> | <填> | <填> |
| ns_tune5_r2_microblog | r2 | -0.3 | microblog | <填> | <填> | <填> | <填> |
| ns_tune5_r3_jobs | r3 | +0.6 | jobs | <填> | <填> | <填> | <填> |
| ns_tune5_r3_congressional | r3 | +0.6 | congressional | <填> | <填> | <填> | <填> |
| ns_tune5_r3_forums | r3 | +0.6 | forums | <填> | <填> | <填> | <填> |
| ns_tune5_r3_microblog | r3 | +0.6 | microblog | <填> | <填> | <填> | <填> |
| ns_tune5_r4_jobs | r4 | -0.6 | jobs | <填> | <填> | <填> | <填> |
| ns_tune5_r4_congressional | r4 | -0.6 | congressional | <填> | <填> | <填> | <填> |
| ns_tune5_r4_forums | r4 | -0.6 | forums | <填> | <填> | <填> | <填> |
| ns_tune5_r4_microblog | r4 | -0.6 | microblog | <填> | <填> | <填> | <填> |

## 2. 与 PrE-Text 基线对比（best_top1）

| 数据集 | PrE-Text | r1 (+0.3) | r2 (-0.3) | r3 (+0.6) | r4 (-0.6) | 是否任一组超过 PrE-Text |
|---|---|---|---|---|---|---|
| jobs | 0.2732 | <填> | <填> | <填> | <填> | <Y/N> |
| congressional | 0.2950 | <填> | <填> | <填> | <填> | <Y/N> |
| forums | 0.2501 | <填> | <填> | <填> | <填> | <Y/N> |
| microblog | 0.2763 | <填> | <填> | <填> | <填> | <Y/N> |

## 3. 长度调制方向性结论

对比 r1+r3（α>0：长保护）与 r2+r4（α<0：短保护）的均值：

| 方向 | jobs 均值 | congressional 均值 | forums 均值 | microblog 均值 |
|---|---|---|---|---|
| α>0 (r1+r3 平均) | <填> | <填> | <填> | <填> |
| α<0 (r2+r4 平均) | <填> | <填> | <填> | <填> |

**结论**：哪个方向（长保护 / 短保护）更有效？是否各数据集偏好一致？

## 4. forums 顽疾是否突破

| 指标 | PrE-Text | Round 4 g1 最佳 | Round 5 r1-r4 最佳 |
|---|---|---|---|
| best_top1 | 0.2501 | 0.2500 | <填> |
| best_top3 | 0.3877 | 0.3856 | <填> |
| best_top5 | 0.4548 | 0.4488 | <填> |
| best_top10 | 0.5375 | 0.5376 | <填> |

forums 在 Round 5 r1-r4 中是否首次破 PrE-Text 在 top1 上？top3/5 上呢？

## 5. 方向 2a 成功判据评估（按 spec §3.8）

- **强成功**：r1-r4 中存在某一组，4 个数据集 best_top1 全部超过 PrE-Text → <Y/N>
- **方向性成功**：长保护 vs 短保护方向显著（相差 > 0.005）→ <Y/N>，方向：<填长/短>
- **失败但有用**：r1-r4 全部不超过 g1 baseline → <Y/N>，作为 ablation 价值

## 6. 关键发现（3-5 条）

（基于实际数据，写出本轮的核心发现，包括有效组合、无效组合、出乎意料的结果）

## 7. Round 5 整体收尾建议

结合方向 1（g5/g6/g7）的结果，给出最终交付配置建议：
- 单一全局配置：是 / 否，哪个？
- per-dataset 最优组合：jobs=<g_or_r_X>, congressional=<X>, forums=<X>, microblog=<X>
- 是否需要 Round 6？如需，方向是什么？
```

- [ ] **Step 4: 填入 Step 2 输出的实际数值**

把 16 个实验数值对照模板填入；double-check 实验 ID 与数据对应。

- [ ] **Step 5: Commit**

```bash
cd /Users/apple/Desktop/code_from_paper
git add paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md
git commit -m "docs(round5): add r1-r4 experiment results with cross-version comparison"
```

---

## Task 20: 整体收尾验证

- [ ] **Step 1: 三层 sanity 全过的最终确认**

回到本 plan 头部确认：
- (a) 6 个 unit test 全 pass（Task 2-5, 8, 10）：✅
- (b) sanity check 三项一致（Task 16）：✅
- (c) 钳制范围 in_band ≥ 80%（Task 17）或已记录调整：✅

- [ ] **Step 2: 跨方向合并分析**

合并方向 1（g5/g6/g7）和方向 2a（r1-r4）的全部结果，得到 **7 + 4 = 11 个 Round 5 配置 + 原 4 个 Round 4 配置 = 15 个配置 × 4 数据集 = 60 个数据点**。

针对每个数据集，找最优配置：

```
jobs       最优 = ? from {g1..g7, r1..r4}
congressional 最优 = ? from {g1..g7, r1..r4}
forums     最优 = ? from {g1..g7, r1..r4}
microblog  最优 = ? from {g1..g7, r1..r4}
```

判断：
- 是否存在**单一**配置在 4 个数据集上同时超过 PrE-Text？→ 论文主推这个
- 否则：per-dataset 最优组合是否能覆盖 4 个？→ 论文用 per-dataset 表

- [ ] **Step 3: 写入最终结论**

在 `paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md` §7 填入。

- [ ] **Step 4: Commit**

```bash
git add paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md
git commit -m "docs(round5): add final cross-direction analysis and configuration recommendation"
```

---

## Self-Review

- [ ] **Spec coverage**：本 plan 是否覆盖了 spec §3 全部要求？
  - §3.1 数学定义 → Task 2-5（test 1-4 验证公式）+ Task 6-7（实现）
  - §3.2 代码改动 → Task 1（copy）, Task 6 (genericity 单条), Task 7 (genericity 批量), Task 9 (stage1_runner), Task 11 (base config)
  - §3.3 r1-r4 配置 → Task 11 + Task 13
  - §3.4 边界与异常 → Task 5 (test 4 验证钳制)
  - §3.4.1 离线分布检查 → Task 17
  - §3.5 6 个测试 → Task 2 (test 1), Task 3 (test 2), Task 4 (test 3), Task 5 (test 4), Task 8 (test 5), Task 10 (test 6)
  - §3.6 sanity check → Task 12 (r0 config), Task 16 (执行 + 三层比对)
  - §3.7 自动化执行 → Task 14 (runner), Task 15 (上传), Task 18 (启动)
  - §3.8 成功判据 → Task 19 §5
  - §3.9 文档输出 → Task 19, Task 20
  - **覆盖完整。**
- [ ] **Placeholder scan**：搜索 `TODO`、`TBD`、`<填>`等
  - 在 Task 19/20 中存在 `<填>` `<Y/N>` 占位 — 这是**模板占位**（运行后由实际数据填入），plan 已注明每个占位的来源（Step 2 输出）
  - 注：Task 16 Step 2 中提到的 `selected_texts.json` 文件名是基于 spec 推测，如远端实际文件名不同，需在 Task 16 内调整 — Plan 已加说明
- [ ] **Type consistency**：函数签名/参数名一致
  - `compute_length_factors`: keyword-only，参数名 `lengths`, `alpha`, `l_ref_strategy`, `factor_min`, `factor_max` — Task 2 实现，Task 3-5 测试调用，全部一致
  - `compute_genericity_penalty` 新增参数：`candidate_length`, `l_ref`, `length_modulation_enabled`, `length_alpha`, `length_factor_min`, `length_factor_max` — Task 6 定义，Task 7 调用一致
  - `compute_genericity_penalties` 新增参数：`candidate_lengths`（注意复数！）, `length_modulation_enabled`, `length_alpha`, `length_factor_min`, `length_factor_max` — Task 7 定义，Task 9 stage1_runner 调用，Task 10 test 验证 kwargs，全部一致
  - **类型一致。**

---

## 完成判定

本 plan 完成的标志：
1. 6 个 unit test 全部 pass，覆盖 alpha=0 等价、长保护、短保护、钳制、disabled 等价 Round 4、stage1_runner 透传
2. r0 sanity check 三层（selected_texts、hard_negative_texts、best_top1）全部通过
3. 离线钳制范围分布检查 in_band ≥ 80%（或已显式接受/调整）
4. 16 个 r1-r4 实验全部 SUCCESS
5. `paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md` 包含完整结果与 per-dataset 最优分析
6. Round 5 整体（方向 1 + 方向 2a）的最终交付配置建议已写入

预计总时长：**3-4 小时**（其中代码改动 ~30 分钟，测试 ~10 分钟，sanity check ~15 分钟，离线分布检查 ~5 分钟，正式实验 ~80 分钟，文档整理 ~30 分钟）。
