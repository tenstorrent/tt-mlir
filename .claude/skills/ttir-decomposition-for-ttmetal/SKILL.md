---
name: ttir-decomposition-for-ttmetal
description: >-
  Add a new TTIR op decomposition pass for the TTMetal pipeline. Use when the
  user wants to decompose/lower a high-level TTIR op (e.g. rms_norm, sdpa,
  layer_norm) into primitive TTIR ops (matmul, add, multiply, softmax, etc.)
  for the D2M/TTMetal backend. Also trigger when the user mentions
  "decomposition pass", "decompose op for ttmetal", or "lower op to primitives".
---

# TTIR Op Decomposition for TTMetal

Decompose a high-level fused TTIR op into primitive TTIR ops so the
D2M/TTMetal backend can lower them individually. The TTNN backend keeps
native fused ops; these decomposition passes only run in the TTMetal pipeline.

## When to Use

- The op has no D2M/TTMetal lowering but can be expressed as a sequence of ops
  that do (matmul, add, multiply, softmax, reduce, reshape, permute, etc.).
- The TTNN backend already has native support, so it skips the decomposition.

## Files to Create / Modify

| File | Action |
|------|--------|
| `lib/Dialect/TTIR/Transforms/<OpName>Decomposition.cpp` | **Create** — the pass implementation |
| `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` | **Edit** — register the pass definition |
| `lib/Dialect/TTIR/Transforms/CMakeLists.txt` | **Edit** — add the new .cpp |
| `lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp` | **Edit** — add pass to frontend pipeline |

## Step-by-Step

### 1. Understand the Op

Read the op definition in `include/ttmlir/Dialect/TTIR/IR/TTIROps.td`.
Note tensor shapes, attributes (optional mask, scale, etc.), and the
mathematical decomposition into primitives.

### 2. Create the Decomposition Pass

Create `lib/Dialect/TTIR/Transforms/<OpName>Decomposition.cpp`.

**Template** (follow exactly):

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIR<PASSNAME_UPPER>
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static void decompose<OpName>(<OpName>Op op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  // ... decomposition logic ...
  rewriter.replaceOp(op, result);
}

class TTIR<OpName>Decomposition
    : public impl::TTIR<OpName>DecompositionBase<TTIR<OpName>Decomposition> {
public:
  using impl::TTIR<OpName>DecompositionBase<
      TTIR<OpName>Decomposition>::TTIR<OpName>DecompositionBase;

  void runOnOperation() final {
    llvm::SmallVector<<OpName>Op> opsToDecompose;
    getOperation()->walk([&](<OpName>Op op) {
      opsToDecompose.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (<OpName>Op op : opsToDecompose) {
      rewriter.setInsertionPoint(op);
      decompose<OpName>(op, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
```

Key conventions:
- The `#define GEN_PASS_DEF_TTIR...` macro name must match the pass name in
  `Passes.td` (all-caps, no underscores between words).
- Use `IRRewriter`, not `PatternRewriter` — this is a direct walk, not a
  greedy pattern rewrite.
- Collect ops first, then decompose (avoid iterator invalidation).
- Call `rewriter.replaceOp(op, result)` at the end to replace the original.

### 3. Creating TTIR Ops in Decomposition Code

Common op creation patterns (use `rewriter.create<T>(...)`):

```cpp
// Elementwise binary (add, multiply, subtract, etc.)
auto add = rewriter.create<AddOp>(loc, resultType, lhs, rhs);

// MatmulOp with optional transpose
auto mm = rewriter.create<MatmulOp>(loc, resultType, a, b,
    /*transpose_a=*/rewriter.getBoolAttr(false),
    /*transpose_b=*/rewriter.getBoolAttr(true));

// SoftmaxOp
auto sm = rewriter.create<SoftmaxOp>(loc, resultType, input,
    rewriter.getSI32IntegerAttr(dim),
    rewriter.getBoolAttr(false));

// FullOp (scalar constant broadcast to shape)
auto full = rewriter.create<FullOp>(loc, resultType,
    rewriter.getF32FloatAttr(value));

// ReshapeOp
SmallVector<int32_t> shapeI32(newShape.begin(), newShape.end());
auto reshape = rewriter.create<ReshapeOp>(loc, newType, input,
    rewriter.getI32ArrayAttr(shapeI32));

// RepeatOp
auto repeat = rewriter.create<RepeatOp>(loc, repeatedType, input,
    rewriter.getDenseI64ArrayAttr(repeatDims));

// PermuteOp
auto permute = rewriter.create<PermuteOp>(loc, permutedType, input,
    rewriter.getDenseI64ArrayAttr(permutation));

// MeanOp (reduction)
auto mean = rewriter.create<MeanOp>(loc, reducedType, input,
    rewriter.getBoolAttr(/*keep_dim=*/true),
    rewriter.getI32ArrayAttr(reduceDims));

// RsqrtOp (unary)
auto rsqrt = rewriter.create<RsqrtOp>(loc, type, input);
```

### 4. Register in Passes.td

Add to `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` near the
existing decomposition passes:

```tablegen
def TTIR<OpName>Decomposition: Pass<"ttir-<op-name>-decomposition", "::mlir::ModuleOp"> {
  let summary = "Decompose ttir.<op_name> into <list primitive ops>.";
  let description = [{
    Lowers `ttir.<op_name>` to primitive ops: ...

    Used by the TTMetal pipeline before `ttir-rank-normalization` so the
    expanded ops participate in rank normalization like other TTIR ops.
    Backends with a native fused <op_name> (e.g. TTNN) do not need this pass.
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}
```

### 5. Add to CMakeLists.txt

Add the new `.cpp` to `lib/Dialect/TTIR/Transforms/CMakeLists.txt`
(alphabetical order in the source list).

### 6. Wire into TTMetal Pipeline

In `lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp`, add the pass in
`createTTIRToTTMetalFrontendPipeline` after the existing decomposition
passes and **before** `ttir-rank-normalization`:

```cpp
pm.addPass(ttir::createTTIRRMSNormDecomposition());
pm.addPass(ttir::createTTIRSDPADecomposition());
pm.addPass(ttir::createTTIR<OpName>Decomposition());  // <-- new
pm.addPass(ttir::createTTIRExplicateTMs());
```

### 7. Build and Iterate

Run `./build_and_test.sh` or:

```bash
source env/activate
cmake --build build
```

Fix compilation errors, then test with a small MLIR snippet:

```bash
build/bin/ttmlir-opt --ttir-<op-name>-decomposition test_snippet.mlir
```

## Reference Implementations

- **RMSNorm**: `lib/Dialect/TTIR/Transforms/RMSNormDecomposition.cpp`
  Decomposes `rms_norm(x, w, b, eps)` into `x^2 -> mean -> +eps -> rsqrt -> *x -> *w -> +b`.

- **SDPA**: `lib/Dialect/TTIR/Transforms/SDPADecomposition.cpp`
  Decomposes `scaled_dot_product_attention(Q, K, V, mask)` into
  `Q @ K^T -> *scale -> +mask -> softmax -> @ V`, with GQA head expansion
  via reshape+repeat+reshape.

- **MinReduction**: `lib/Dialect/TTIR/Transforms/DecomposeMinReduction.cpp`
  Decomposes `min(x)` into `neg(max(neg(x)))`.
