---
name: ttir-decomposition-for-ttmetal
description: >-
  Add a new composite op decomposition pattern to the TTMetal pipeline. Use
  when the user wants to decompose/lower a high-level TTIR op (e.g. rms_norm,
  sdpa, layer_norm, softmax) into primitive TTIR ops (matmul, add, multiply,
  etc.) for the D2M/TTMetal backend. Also trigger when the user mentions
  "decomposition pattern", "decompose op for ttmetal", or "lower op to
  primitives".
---

# TTIR Composite Op Decomposition for TTMetal

Decompose a high-level fused TTIR op into primitive TTIR ops so the
D2M/TTMetal backend can lower them individually. The TTNN backend keeps
native fused ops; these decomposition patterns only run in the TTMetal
pipeline via the unified `TTIRDecomposeComposites` pass.

## When to Use

- The op has no D2M/TTMetal lowering but can be expressed as a sequence of ops
  that do (matmul, add, multiply, reduce, reshape, permute, etc.).
- The TTNN backend already has native support, so it skips the decomposition.

## Architecture

All composite decompositions live in a single pass
(`ttir-decompose-composites`) that uses MLIR's greedy pattern rewriter. Each
op decomposition is an `OpRewritePattern<T>` with a configurable benefit
level that controls application order. For example, SDPA has higher benefit
than softmax so it runs first — the softmax ops it produces are then caught
by the softmax pattern on subsequent rewriter iterations.

## Files to Modify

| File | Action |
|------|--------|
| `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp` | **Edit** — add a new `OpRewritePattern` |
| `include/ttmlir/Dialect/TTIR/Transforms/Passes.td` | **Edit** — update description if desired |
| `test/ttmlir/Dialect/TTIR/Transforms/metal_composite_decompositions.mlir` | **Edit** — add FileCheck tests |
| `test/python/golden/test_metal_composite_ops.py` | **Edit** — add Python builder tests |

You should NOT need to touch `CMakeLists.txt` or `Passes.td` pass
registration in the common case. However, when adding a new composite
decomposition, verify that `ttir-decompose-composites` is scheduled in the
relevant TTMetal pipeline in `TTMetalPipelines.cpp`, and update that pipeline
if necessary.

## Step-by-Step

### 1. Understand the Op

Read the op definition in `include/ttmlir/Dialect/TTIR/IR/TTIROps.td`.
Note tensor shapes, attributes (optional mask, scale, etc.), and the
mathematical decomposition into primitives.

### 2. Add a Pattern to DecomposeComposites.cpp

Open `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp` and add a new
`OpRewritePattern<YourOp>` struct. Follow the existing patterns as examples.

**Pattern template:**

```cpp
struct DecomposeYourOpPattern : public OpRewritePattern<YourOp> {
  using OpRewritePattern<YourOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(YourOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // ... decomposition logic using rewriter.create<T>(...) ...
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

Then register the pattern in `TTIRDecomposeComposites::runOnOperation()`:

```cpp
void runOnOperation() final {
  RewritePatternSet patterns(&getContext());
  patterns.add<DecomposeSDPAPattern>(&getContext(), /*benefit=*/2);
  patterns.add<DecomposeRMSNormPattern>(&getContext(), /*benefit=*/1);
  patterns.add<DecomposeSoftmaxPattern>(&getContext(), /*benefit=*/0);
  patterns.add<DecomposeYourOpPattern>(&getContext(), /*benefit=*/N);  // NEW

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
```

**Benefit ordering:** If your decomposition produces ops that another pattern
needs to decompose further (e.g. SDPA produces softmax), give your pattern a
higher benefit number than the downstream pattern.

Key conventions:
- Use `OpRewritePattern<T>` and `PatternRewriter`, not `IRRewriter`.
- Always `return success()` after `rewriter.replaceOp(op, result)`.
- Call `rewriter.replaceOp(op, result)` at the end to replace the original.

### 3. Creating TTIR Ops in Decomposition Code

Common op creation patterns (use `rewriter.create<T>(...)`):

```cpp
// Elementwise binary (add, multiply, subtract, etc.)
auto add = rewriter.create<AddOp>(loc, resultType, lhs, rhs);

// MatmulOp
auto mm = rewriter.create<MatmulOp>(loc, resultType, a, b);

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

### 4. Add MLIR Lit Tests

Add test functions and FileCheck assertions to
`test/ttmlir/Dialect/TTIR/Transforms/metal_composite_decompositions.mlir`.

The file uses a single pass (`--ttir-decompose-composites`) with multiple
check prefixes. Add a new check prefix for your op and a new RUN line:

```
// RUN: ttmlir-opt --ttir-decompose-composites %s | FileCheck %s --check-prefix=YOUROP
```

Then add test functions:

```
// YOUROP-LABEL: func.func @your_op_basic
// YOUROP-NOT: ttir.your_op
// YOUROP: "ttir.multiply"
// YOUROP: "ttir.add"
// YOUROP: return
func.func @your_op_basic(%input: tensor<...xbf16>) -> tensor<...xbf16> {
  %0 = "ttir.your_op"(%input) <{...}> : (...) -> ...
  return %0 : ...
}
```

### 5. Add Python Builder Tests

Add tests to `test/python/golden/test_metal_composite_ops.py`. This file
contains all composite decomposition tests for the TTMetal pipeline.

Follow the existing patterns (SDPA, RMSNorm, softmax) as examples:

```python
@pytest.mark.parametrize("shape", [...])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_your_op_decomposition(
    shape: Shape,
    target: str,
    request,
    device,
):
    """Test your_op decomposition for the TTMetal pipeline."""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def your_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.your_op(in0, ..., unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )
```

Key points:
- Always use `target="ttmetal"` — decompositions only run in the TTMetal
  pipeline.
- Use `compile_and_execute_ttir` from `builder.base.builder_apis`.
- Parametrize over the attributes that affect decomposition logic
  (e.g. causal vs non-causal, with/without weight, different dims).

### 6. Build and Iterate

Run `./build_and_test.sh` or:

```bash
source env/activate
cmake --build build
```

Fix compilation errors, then test with the lit test:

```bash
build/bin/ttmlir-opt --ttir-decompose-composites test/ttmlir/Dialect/TTIR/Transforms/metal_composite_decompositions.mlir
```

And the Python tests:

```bash
pytest -svv test/python/golden/test_metal_composite_ops.py
```

## Reference Implementations

All live in `lib/Dialect/TTIR/Transforms/DecomposeComposites.cpp`:

- **DecomposeRMSNormPattern** (benefit 1):
  Decomposes `rms_norm(x, w, b, eps)` into
  `x^2 -> mean -> +eps -> rsqrt -> *x -> *w -> +b`.

- **DecomposeSDPAPattern** (benefit 2):
  Decomposes `scaled_dot_product_attention(Q, K, V, mask)` into
  `Q @ K^T -> *scale -> +mask -> softmax -> @ V`, with GQA head expansion
  via reshape. Produces `ttir.softmax` ops that the softmax pattern then
  decomposes.

- **DecomposeSoftmaxPattern** (benefit 0):
  Decomposes `softmax(x, dim)` into
  `max -> subtract -> exp -> sum -> div` (uses `ttir.div` rather than
  `reciprocal -> multiply` to work around a broadcast-multiply bug).
  When `numericStable=false`, the max-subtract step is skipped.

- **DecomposeMinReduction** (separate pass):
  `lib/Dialect/TTIR/Transforms/DecomposeMinReduction.cpp` — Decomposes
  `min(x)` into `neg(max(neg(x)))`. This is a standalone pass, not part
  of the composites pass.
