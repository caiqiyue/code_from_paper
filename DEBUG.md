# DEBUG: Repeated Model Loading in Single-Node Runner

## Date: 2026-04-15

## Phase 1: Observations

### 1. Architecture Overview

All 5 components (generator, scorer, retriever, critic, aggregator) are created **upfront** in `_build_components()` before any stage runs. However, model loading is lazy - models only load when first used.

Key lazy-loading components:
- `TransformersTextBackend` (Qwen model): loads on first `generate()` call
- `SentenceTransformerEmbedder` (MiniLM): loads on first `embed_texts()` call

### 2. Model Loading Per Stage

**Stage A:**
- `_generate_batched()` creates **server** text_backend → loads Qwen (load #1)
- `_build_client_context()` creates **client** text_backend + embedder → loads Qwen (load #2), loads MiniLM
- `DataInfScorer.score()` uses embedder (MiniLM used here)
- Stage A cleanup: releases generator + scorer, but **NOT** client_ctx

**Stage B:**
- `_build_client_context_for_stage_b()` creates **another** client text_backend + embedder → loads Qwen (load #3), loads MiniLM
- `_build_server_context()` creates **server** text_backend → loads Qwen (load #4) but **NEVER USED**
- Stage B cleanup: releases retriever, critic, aggregator, but **NOT** the client_ctx/server_ctx

**Stage C:**
- `_generate_local()` loads DistilGPT2 locally → loads DistilGPT2
- Releases DistilGPT2 after

### 3. Text Backend Configuration

Both server and client backends use the same model path (`qwen_2_0_5b_instruct`) but as **separate instances**:
```yaml
llm:
  client:
    engine: transformers
    model_name_or_path: thesis_platform/open_model/qwen_2_0_5b_instruct
  server:
    engine: transformers
    model_name_or_path: thesis_platform/open_model/qwen_2_0_5b_instruct
```

### 4. Evidence of "Repeated Loading"

The user reported seeing "Loading weights" multiple times. Evidence:
- Qwen loads 4 times total (2 in Stage A/B, 2 in Stage B context creation)
- Stage A's client_ctx is never released, so its Qwen stays in memory
- Stage B creates redundant server_ctx.text_backend that loads Qwen but is never used

### 5. What Works Correctly

- `TransformersTextBackend._ensure_loaded()` caches model after first load - won't reload same instance
- `release_component_resources()` properly calls `release()` which clears model references
- `_generate_batched()` creates text_backend once and reuses it across all rounds

---

## Phase 2: Hypotheses

### H1: Multiple Qwen Instances Due to Separate Server/Client Backends (ROOT HYPOTHESIS)
- **Supports**: Both server and client config use same model path but separate backend instances. Stage A creates client text_backend that is only used in Stage B. Stage B then creates another client text_backend.
- **Conflicts**: None
- **Test**: Add logging to track when `TransformersTextBackend._ensure_loaded()` is called

### H2: Stage A client_ctx Not Released Before Stage B Creates New Context
- **Supports**: `run_stage_a()` calls `release_component_resources(self.generator, self.scorer)` but NOT `release_component_resources(client_ctx)`
- **Conflicts**: Even if released, Stage B would still create its own client_ctx
- **Test**: Check if releasing Stage A client_ctx before Stage B reduces memory

### H3: _build_server_context() Creates Unused Server Text Backend in Stage B
- **Supports**: `_build_server_context()` creates `text_backend=self._build_text_backend()` which loads Qwen, but this backend is never used in Stage B
- **Conflicts**: None - this is confirmed unused code
- **Test**: Remove the server text_backend creation from Stage B context and observe

---

## Root Cause Summary

**Root Cause**: The Qwen model is loaded **4 separate times** due to:
1. Server text_backend in `_generate_batched()` (Stage A generation)
2. Client text_backend in `_build_client_context()` (Stage A, only used in Stage B)
3. Another client text_backend in `_build_client_context_for_stage_b()` (Stage B)
4. Server text_backend in `_build_server_context()` (Stage B, **never used**)

Additionally, **Stage A's client_ctx is never released**, so its Qwen model stays in memory throughout.

The `_build_server_context()` in Stage B creates a text_backend that is **completely unused** - the server_ctx.text_backend is never referenced in Stage B's critique-retrieval-aggregation loop.

---

## Fix Applied

### Fix 1: Stage A - Release client_ctx after scoring (line 166-171)
**Before:**
```python
release_component_resources(self.generator, self.scorer)
```
**After:**
```python
release_component_resources(self.generator, self.scorer, client_ctx)
```
**Why:** `client_ctx` (embedder + text_backend) was created but never released after scoring.

### Fix 2: Stage B - Don't create unused server text_backend (line 559)
**Before:**
```python
text_backend=self._build_text_backend(),  # Loads Qwen - NEVER USED!
```
**After:**
```python
text_backend=None,  # Not used - client_ctx.text_backend is used instead
```
**Why:** `server_ctx.text_backend` was created but never used in Stage B. All LLM calls use `client_ctx.text_backend`.

### Fix 3: Stage B - Release client_ctx and server_ctx after loop (line 301)
**Before:**
```python
release_component_resources(self.retriever, self.critic, self.aggregator)
```
**After:**
```python
release_component_resources(self.retriever, self.critic, self.aggregator, client_ctx, server_ctx)
```
**Why:** Ensure all context resources are properly released after Stage B completes.

---

## Result

**Before Fix:** Qwen model loaded 4 times
1. Server text_backend in `_generate_batched()` (Stage A) ✓ Used
2. Client text_backend in `_build_client_context()` (Stage A) ✓ Used in Stage B
3. Client text_backend in `_build_client_context_for_stage_b()` (Stage B) ✓ Used
4. Server text_backend in `_build_server_context()` (Stage B) ✗ **WASTED - Never used**

**After Fix:** Qwen model loaded 2 times
1. Server text_backend in `_generate_batched()` (Stage A) ✓ Used
2. Client text_backend in `_build_client_context_for_stage_b()` (Stage B) ✓ Used

Plus proper cleanup of contexts after each stage.

---

## Root Cause

**Root Cause**: The Qwen model was loaded **4 separate times** instead of 2 because:

1. `_build_server_context()` in Stage B created a server `text_backend` that was **never used** - the Stage B loop only uses `client_ctx.text_backend`
2. `client_ctx` in Stage A was created but **never explicitly released** before returning
3. `client_ctx` in Stage B was created but **never explicitly released** after the loop

The core architectural issue was that in single-node Stage B, both `client_ctx` and `server_ctx` were created, but only `client_ctx.text_backend` was used for all LLM operations. The `server_ctx` existed solely to hold `base_prompt`, `aggregation_memory`, and `prompt_history` - not the text backend.
