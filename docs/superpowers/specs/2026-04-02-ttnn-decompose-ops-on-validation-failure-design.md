# TTNNDecomposeOpsOnValidationFailure Pass Design

## Overview

A new generic TTNN pass that validates operations against device constraints using
`FusionValidator` and decomposes them to equivalent component operations when
validation fails. SDPA is the first operation supported, with the pass designed
to accept additional decomposition patterns over time.

## Motivation

SDPA operations entering the TTNN pipeline via TTIR-to-TTNN conversion are not
validated against device constraints. If such an op cannot execute on device
(e.g., unsupported head dimension, shape constraints), the compiler currently
has no fallback path. This pass provides that fallback by decomposing the op
into simpler component operations that the device can execute individually.

## Pass Structure

**Name:** `TTNNDecomposeOpsOnValidationFailure`

**Type:** `Pass<"ttnn-decompose-ops-on-validation-failure", "::mlir::ModuleOp">`

**Approach:** Register `RewritePattern` subclasses per op type, run via
`applyPatternsGreedily` with top-down traversal. Each pattern:
1. Calls `FusionValidator::validateFusion` on the matched op.
2. If validation **succeeds** -> return `failure()` (no rewrite needed).
3. If validation **fails** -> decompose the op into component TTNN ops.

**Options:**
- `enableOpConstraints` (bool, default: false) — Enable op model constraint
  validation (requires TTMLIR_ENABLE_OPMODEL).
- `maxFallbackAttempts` (uint32_t, default: 10000) — Max fallback configs to
  try before giving up.
- `forceDecompose` (bool, default: false) — Skip FusionValidator and always
  decompose. Used for testing decomposition patterns without requiring OpModel.

**Pipeline Placement:** Immediately after `TTNNFusing`, wrapped in
`DevicePassesWrapper` (since `FusionValidator` needs `system_desc`). The
`forceDecompose` mode does not require `DevicePassesWrapper`.

**Rationale for placement:** `FusionValidator` internally applies workaround
passes (decomposition + layout workarounds) during validation. If SDPA still
fails after workarounds, it truly cannot run on device.

## SDPA Decomposition — Layered Approach

Decomposition is layered: SDPADecode decomposes to regular SDPA first, then
regular SDPA decomposes to component ops if it also fails validation. The
greedy driver naturally cascades these.

### Pattern 1: SDPADecode -> SDPA

**Matches:** `ttnn.scaled_dot_product_attention_decode`

**Decomposition:**
1. Permute Q: `[1, B, H, D]` -> `[B, H, 1, D]`
2. Create `ttnn.scaled_dot_product_attention` with permuted Q, carrying over
   `attention_sink`, `attention_mask`, `scale`, `is_causal`.
3. Unpermute result: `[B, H, 1, D]` -> `[1, B, H, D]`

**Dropped attributes:** `cur_pos_tensor` (KV cache index, not needed for
decomposed attention), `program_config`.

### Pattern 2: SDPA -> Component Ops

**Matches:** `ttnn.scaled_dot_product_attention`

**Decomposition steps:**

#### Step 1: GQA Head Expansion
If `num_heads != num_kv_heads`:
- `ttnn.repeat_interleave(K, repeats=num_heads/num_kv_heads, dim=1)`
- `ttnn.repeat_interleave(V, repeats=num_heads/num_kv_heads, dim=1)`

#### Step 2: Transpose K
- `ttnn.transpose(K, dim0=-2, dim1=-1)`: `[B, H, Skv, D]` -> `[B, H, D, Skv]`

#### Step 3: Matmul Q @ K^T
- `ttnn.matmul(Q, K_transposed)` -> scores `[B, H, Sq, Skv]`

#### Step 4: Scale
Always applied explicitly after matmul (component ops have no implicit scaling):
- If `scale` attr present: `ttnn.multiply(scores, scale)`
- If `scale` attr absent: `ttnn.multiply(scores, 1.0/sqrt(head_dim))`

#### Step 5: Mask
- If `attention_mask` present: `ttnn.add(scores, mask)`
- If `is_causal=true` (no explicit mask): generate causal mask, then add.
  See "Causal Mask Generation" below.
- If `sliding_window_size` set: generate sliding window mask. See "Sliding
  Window Mask Generation" below.

#### Step 6: Attention Sink (if present)
- `ttnn.concat(scores, attention_sink, dim=-1)`:
  `[B, H, Sq, Skv] + [B, H, Sq, sink_width]` -> `[B, H, Sq, Skv+sink_width]`

No 1/scale workaround undo is needed. SDPA ops reaching this pass come from
TTIR-to-TTNN conversion where the attention_sink is unmodified (the 1/scale
compensation is only applied during TTNNFusing).

#### Step 7: Softmax
- `ttnn.softmax(scores, dim=-1)`

#### Step 8: Slice (if attention_sink was present)
- Slice on last dimension to remove sink columns:
  `[B, H, Sq, Skv+sink_width]` -> `[B, H, Sq, Skv]`

#### Step 9: Matmul scores @ V
- `ttnn.matmul(softmax_out, V)` -> result `[B, H, Sq, D]`

## Causal Mask Generation

When `is_causal=true` and no explicit mask is provided, generate a causal mask
of shape `[1, 1, Sq, Skv]`:
- Build row indices via `arange(0, Sq)` and column indices via `arange(0, Skv)`
- Compare: `rows >= cols` produces the lower-triangular boolean mask
- Use `where(condition, 0.0, -inf)` to create the additive mask

## Sliding Window Mask Generation

When `sliding_window_size` is set, generate a mask where positions outside
`[pos - window_size, pos]` are `-inf`. If combined with `is_causal=true`,
intersect with causal mask (both constraints apply).

## File Layout

```
include/ttmlir/Dialect/TTNN/Transforms/Passes.td
  -> Add TTNNDecomposeOpsOnValidationFailure pass definition

lib/Dialect/TTNN/Transforms/TTNNDecomposeOpsOnValidationFailure.cpp
  -> Pass implementation, pattern registration

lib/Dialect/TTNN/Transforms/Decomposition/
  -> Directory for decomposition patterns (mirrors Fusing/ structure)

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp
  -> SDPADecode -> SDPA pattern

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp
  -> SDPA -> component ops pattern

lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp
  -> Register pass after TTNNFusing in pipeline

test/ttmlir/Dialect/TTNN/decompose/
  -> Lit tests using forceDecompose flag
```

## Scope

**In scope:**
- `ScaledDotProductAttentionDecodeOp` decomposition to `ScaledDotProductAttentionOp`
- `ScaledDotProductAttentionOp` decomposition to component TTNN ops
- All SDPA variants: MHA, MQA, GQA
- Attention sink handling (concat before softmax, slice after)
- Causal mask generation
- Sliding window mask generation
- `forceDecompose` testing flag

**Out of scope (future work):**
- `PagedScaledDotProductAttentionDecodeOp`
- Other op decomposition patterns beyond SDPA
