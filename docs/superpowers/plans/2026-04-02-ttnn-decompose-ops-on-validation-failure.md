# TTNNDecomposeOpsOnValidationFailure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic TTNN pass that validates ops via FusionValidator and decomposes them to component ops when validation fails, starting with SDPA.

**Architecture:** RewritePattern-based pass (like TTNNFusing) placed after TTNNFusing in the pipeline. Two layered patterns: SDPADecode->SDPA, then SDPA->component ops. A `forceDecompose` flag enables testing without OpModel.

**Tech Stack:** MLIR C++, TableGen, lit/FileCheck tests

**Spec:** `docs/superpowers/specs/2026-04-02-ttnn-decompose-ops-on-validation-failure-design.md`

---

## File Structure

```
include/ttmlir/Dialect/TTNN/Transforms/Passes.td
  -> Add pass definition (modify)

lib/Dialect/TTNN/Transforms/TTNNDecomposeOpsOnValidationFailure.cpp
  -> Pass implementation + pattern registration (create)

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h
  -> SDPADecode->SDPA pattern header (create)

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp
  -> SDPADecode->SDPA pattern implementation (create)

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h
  -> SDPA->component ops pattern header (create)

lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp
  -> SDPA->component ops pattern implementation (create)

lib/Dialect/TTNN/Transforms/CMakeLists.txt
  -> Register new source files (modify)

lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp
  -> Register pass after TTNNFusing (modify)

test/ttmlir/Dialect/TTNN/decompose/sdpa_decode_decompose.mlir
  -> SDPADecode decomposition tests (create)

test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir
  -> SDPA decomposition tests (create)
```

---

### Task 1: Pass Definition, Skeleton, CMake, and Pipeline

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Transforms/Passes.td:281`
- Create: `lib/Dialect/TTNN/Transforms/TTNNDecomposeOpsOnValidationFailure.cpp`
- Modify: `lib/Dialect/TTNN/Transforms/CMakeLists.txt:28`
- Modify: `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:290`

- [ ] **Step 1: Add pass definition in Passes.td**

Insert after the `TTNNFusing` definition (after line 281) in
`include/ttmlir/Dialect/TTNN/Transforms/Passes.td`:

```tablegen
def TTNNDecomposeOpsOnValidationFailure: Pass<"ttnn-decompose-ops-on-validation-failure", "::mlir::ModuleOp">
{
  let summary = "TTNN decompose ops on validation failure pass.";
  let description = [{
    This pass validates operations against device constraints using
    FusionValidator and decomposes them to equivalent component operations
    when validation fails. This provides a fallback path for ops that
    cannot execute on device in their fused form.
  }];

  let options = [
    Option<"enableOpConstraints",
            "enable-op-constraints",
            "bool", /*default=*/"false",
            "Enable op model constraint validation. "
            "Requires TTMLIR_ENABLE_OPMODEL.">,
    Option<"maxFallbackAttempts",
            "max-fallback-attempts",
            "uint32_t", /*default=*/"10000",
            "Maximum number of fallback configurations to try.">,
    Option<"forceDecompose",
            "force-decompose",
            "bool", /*default=*/"false",
            "Skip validation and always decompose. Used for testing.">
  ];
}
```

- [ ] **Step 2: Create pass implementation skeleton**

Create `lib/Dialect/TTNN/Transforms/TTNNDecomposeOpsOnValidationFailure.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"
#endif

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDECOMPOSEOPSONVALIDATIONFAILURE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNDecomposeOpsOnValidationFailurePass
    : public impl::TTNNDecomposeOpsOnValidationFailureBase<
          TTNNDecomposeOpsOnValidationFailurePass> {
public:
  using impl::TTNNDecomposeOpsOnValidationFailureBase<
      TTNNDecomposeOpsOnValidationFailurePass>::
      TTNNDecomposeOpsOnValidationFailureBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    if (forceDecompose) {
      patterns.add<decomposition::SDPADecodeDecompositionPattern>(
          &getContext());
      patterns.add<decomposition::SDPADecompositionPattern>(&getContext());
    }

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      FusionValidationConfig validationConfig;
      validationConfig.maxFallbackAttempts = maxFallbackAttempts;

      patterns.add<decomposition::SDPADecodeDecompositionPattern>(
          &getContext(), validationConfig);
      patterns.add<decomposition::SDPADecompositionPattern>(
          &getContext(), validationConfig);
    }
#endif

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::tt::ttnn
```

- [ ] **Step 3: Create stub pattern headers**

Create `include/ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECODEDECOMPOSITIONPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECODEDECOMPOSITIONPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

class SDPADecodeDecompositionPattern
    : public OpRewritePattern<ScaledDotProductAttentionDecodeOp> {
public:
  // Constructor for forceDecompose mode (no validation).
  explicit SDPADecodeDecompositionPattern(MLIRContext *context)
      : OpRewritePattern(context), forceDecompose(true) {}

  // Constructor for validation mode.
  SDPADecodeDecompositionPattern(MLIRContext *context,
                                 const FusionValidationConfig &config)
      : OpRewritePattern(context), forceDecompose(false),
        validationConfig(config) {}

  LogicalResult matchAndRewrite(ScaledDotProductAttentionDecodeOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool forceDecompose;
  std::optional<FusionValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif
```

Create `include/ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_SDPADECOMPOSITIONPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

class SDPADecompositionPattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  // Constructor for forceDecompose mode (no validation).
  explicit SDPADecompositionPattern(MLIRContext *context)
      : OpRewritePattern(context), forceDecompose(true) {}

  // Constructor for validation mode.
  SDPADecompositionPattern(MLIRContext *context,
                           const FusionValidationConfig &config)
      : OpRewritePattern(context), forceDecompose(false),
        validationConfig(config) {}

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool forceDecompose;
  std::optional<FusionValidationConfig> validationConfig;
};

} // namespace mlir::tt::ttnn::decomposition

#endif
```

- [ ] **Step 4: Create stub pattern .cpp files**

Create `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult SDPADecodeDecompositionPattern::matchAndRewrite(
    ScaledDotProductAttentionDecodeOp op,
    PatternRewriter &rewriter) const {
  // TODO: Implement in Task 2
  return failure();
}

} // namespace mlir::tt::ttnn::decomposition
```

Create `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult SDPADecompositionPattern::matchAndRewrite(
    ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const {
  // TODO: Implement in Task 3
  return failure();
}

} // namespace mlir::tt::ttnn::decomposition
```

- [ ] **Step 5: Update CMakeLists.txt**

In `lib/Dialect/TTNN/Transforms/CMakeLists.txt`, add after line 28 (`TTNNFusing.cpp`):

```cmake
        TTNNDecomposeOpsOnValidationFailure.cpp
        Decomposition/SDPADecodeDecompositionPattern.cpp
        Decomposition/SDPADecompositionPattern.cpp
```

- [ ] **Step 6: Register pass in pipeline**

In `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`, add a new function before
`createTTNNPipelineWorkaroundPass` (around line 183):

```cpp
// Create the decompose-on-validation-failure pass, placed after fusing.
// When optimizer is enabled, wraps in DevicePassesWrapper for FusionValidator.
// When not, adds with forceDecompose=false (no-op without optimizer).
void createTTNNDecomposeOpsOnValidationFailurePass(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {
  if (options.optimizerPassEnabled) {
#ifdef TTMLIR_ENABLE_OPMODEL
    DevicePassesWrapperOptions wrapperOptions;
    wrapperOptions.devicePtr = options.devicePtr;
    wrapperOptions.tensorL1UsageCap = options.tensorL1UsageCap;

    uint32_t fallbackAttempts = options.maxFallbackAttempts;
    pm.addPass(createDevicePassesWrapper(
        [fallbackAttempts](OpPassManager &innerPm) {
          TTNNDecomposeOpsOnValidationFailureOptions opts;
          opts.enableOpConstraints = true;
          opts.maxFallbackAttempts = fallbackAttempts;
          innerPm.addPass(
              createTTNNDecomposeOpsOnValidationFailure(opts));
        },
        wrapperOptions));
#endif
  }
}
```

Then in the pipeline construction (around line 290), add the call right after
`createTTNNFusingPass`:

```cpp
    createTTNNFusingPass(devicePm, options);
    createTTNNDecomposeOpsOnValidationFailurePass(devicePm, options);
```

- [ ] **Step 7: Build and verify pass exists**

Run:
```bash
source env/activate && cmake --build build -- -j$(nproc) ttmlir-opt 2>&1 | tail -20
```

Then verify the pass is registered:
```bash
build/bin/ttmlir-opt --help | grep decompose-ops-on-validation-failure
```

Expected: pass name appears in help output.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "Add TTNNDecomposeOpsOnValidationFailure pass skeleton

Register pass definition, empty patterns, CMake, and pipeline integration.
Pass runs after TTNNFusing with forceDecompose flag for testing."
```

---

### Task 2: SDPADecode Decomposition Pattern

**Files:**
- Modify: `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp`
- Create: `test/ttmlir/Dialect/TTNN/decompose/sdpa_decode_decompose.mlir`

**References:**
- SDPAFusingPattern creates decode ops with permute: `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp:697-726`
- Permute utility: `lib/Conversion/TTIRToTTNN/Utils.cpp:46-59`
- SDPA decode op creation: `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:3358-3388`
- Existing SDPA decode test IR: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_pad_sequence_dim_workaround.mlir:177-244`

- [ ] **Step 1: Write the lit test**

Create `test/ttmlir/Dialect/TTNN/decompose/sdpa_decode_decompose.mlir`:

```mlir
// RUN: ttmlir-opt --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test 1: Basic SDPADecode MHA decomposition -> SDPA with permutes
  // Q: [1, B, H, D] -> permute to [B, H, 1, D] -> SDPA -> permute back
  func.func @sdpa_decode_mha(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_mha
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = [1, 2, 0, 3]
    // CHECK: %[[SDPA:.*]] = "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2, %arg3)
    // CHECK-SAME: is_causal = false
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK: %[[RESULT:.*]] = "ttnn.permute"(%[[SDPA]])
    // CHECK-SAME: permutation = [2, 0, 1, 3]
    // CHECK: return %[[RESULT]]
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<32x1x1x128xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 2: SDPADecode without mask (causal)
  func.func @sdpa_decode_causal(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_causal
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK-SAME: permutation = [1, 2, 0, 3]
    // CHECK: %[[SDPA:.*]] = "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2)
    // CHECK-SAME: is_causal = true
    // CHECK: %[[RESULT:.*]] = "ttnn.permute"(%[[SDPA]])
    // CHECK-SAME: permutation = [2, 0, 1, 3]
    // CHECK: return %[[RESULT]]
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 3: SDPADecode with attention_sink
  func.func @sdpa_decode_with_attention_sink(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>,
    %sink: tensor<1x32x1x1xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_with_attention_sink
    // CHECK: %[[PERMUTED_Q:.*]] = "ttnn.permute"(%arg0)
    // CHECK: %[[SDPA:.*]] = "ttnn.scaled_dot_product_attention"(%[[PERMUTED_Q]], %arg1, %arg2, %arg3
    // CHECK-SAME: %arg4
    // CHECK: %[[RESULT:.*]] = "ttnn.permute"(%[[SDPA]])
    // CHECK: return %[[RESULT]]
    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask, %sink) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<32x1x1x128xbf16>,
         tensor<1x32x1x1xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/sdpa_decode_decompose.mlir -v 2>&1 | tail -20
```

Expected: FAIL — pattern currently returns `failure()` so nothing is decomposed.

- [ ] **Step 3: Implement SDPADecode decomposition pattern**

Replace the contents of
`lib/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {

// SDPADecode Q shape: [1, B, H, D]
// SDPA Q shape:       [B, H, Sq, D]
// Permutation: [1, 2, 0, 3] maps [1, B, H, D] -> [B, H, 1, D]
static constexpr std::array<int64_t, 4> kToSDPAPermutation = {1, 2, 0, 3};
// Inverse: [2, 0, 1, 3] maps [B, H, 1, D] -> [1, B, H, D]
static constexpr std::array<int64_t, 4> kFromSDPAPermutation = {2, 0, 1, 3};

LogicalResult SDPADecodeDecompositionPattern::matchAndRewrite(
    ScaledDotProductAttentionDecodeOp op,
    PatternRewriter &rewriter) const {

  if (!forceDecompose) {
    FusionValidator validator(rewriter.getContext(), *validationConfig);

    auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());
    auto validationResult =
        validator.validateFusion<ScaledDotProductAttentionDecodeOp>(
            op.getOperation(), op.getLoc(), {qType}, op.getQuery(), op.getKey(),
            op.getValue(), op.getIsCausal(), op.getAttentionMask(),
            op.getCurPosTensor(), op.getAttentionSink(), op.getScaleAttr(),
            op.getMemoryConfigAttr(), op.getProgramConfigAttr());

    if (validationResult.isSuccess()) {
      return failure(); // Op is valid on device, no decomposition needed.
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                 "SDPA decode validation failed, decomposing: {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();

  // Step 1: Permute Q from [1, B, H, D] to [B, H, 1, D].
  Value permutedQuery = ttir_to_ttnn::utils::generatePermute(
      mlir::cast<TypedValue<RankedTensorType>>(op.getQuery()),
      llvm::to_vector(kToSDPAPermutation), rewriter, loc);

  // Step 2: Create ScaledDotProductAttentionOp.
  // Result type matches permuted Q shape: [B, H, 1, D].
  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      loc, permutedQuery.getType(), permutedQuery, op.getKey(), op.getValue(),
      op.getAttentionMask(),
      op.getIsCausal(), op.getScaleAttr(),
      /*sliding_window_size=*/IntegerAttr(), op.getAttentionSink(),
      /*memory_config=*/MemoryConfigAttr());

  // Step 3: Permute result back from [B, H, 1, D] to [1, B, H, D].
  Value finalResult = ttir_to_ttnn::utils::generatePermute(
      sdpaOp.getResult(),
      llvm::to_vector(kFromSDPAPermutation), rewriter, loc);

  rewriter.replaceOp(op, finalResult);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
```

- [ ] **Step 4: Build and run test**

```bash
cmake --build build -- -j$(nproc) ttmlir-opt 2>&1 | tail -5
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/sdpa_decode_decompose.mlir -v
```

Expected: PASS for all three test cases.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Implement SDPADecode decomposition to SDPA

Decomposes ttnn.scaled_dot_product_attention_decode to
ttnn.scaled_dot_product_attention by permuting Q [1,B,H,D]->[B,H,1,D]
and unpermuting the result. Carries over mask, scale, attention_sink."
```

---

### Task 3: SDPA Basic Decomposition (MHA with Mask)

**Files:**
- Modify: `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp`
- Modify: `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`

**References:**
- Op creation patterns: see Task 1 references
- `ttnn::utils::RankedTensorTypeFactory::create()` for type construction
- `ttmlir/Dialect/TTNN/Utils/TransformUtils.h` for TTNN utility functions
- Matmul creation: `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.cpp:166`
- FullOp + MultiplyOp for scaling: `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp:686-691`

- [ ] **Step 1: Write the lit test for basic MHA**

Create `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`:

```mlir
// RUN: ttmlir-opt --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test 1: Basic MHA SDPA with mask and explicit scale
  // Q/K/V: [B=1, H=8, S=64, D=64], mask: [1, 1, 64, 64]
  func.func @sdpa_mha_with_mask(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_mha_with_mask

    // Step 1: Transpose K last two dims [1,8,64,64] -> [1,8,64,64]
    // CHECK: %[[K_T:.*]] = "ttnn.transpose"(%arg1)
    // CHECK-SAME: dim0 = -2 : si32
    // CHECK-SAME: dim1 = -1 : si32

    // Step 2: Matmul Q @ K^T -> [1,8,64,64]
    // CHECK: %[[SCORES:.*]] = "ttnn.matmul"(%arg0, %[[K_T]])

    // Step 3: Scale by 0.125
    // CHECK: %[[SCALE:.*]] = "ttnn.full"
    // CHECK-SAME: fill_value = 1.250000e-01
    // CHECK: %[[SCALED:.*]] = "ttnn.multiply"(%[[SCORES]], %[[SCALE]])

    // Step 4: Add mask
    // CHECK: %[[MASKED:.*]] = "ttnn.add"(%[[SCALED]], %arg3)

    // Step 5: Softmax
    // CHECK: %[[SOFTMAX:.*]] = "ttnn.softmax"(%[[MASKED]])
    // CHECK-SAME: dimension = -1

    // Step 6: Matmul softmax @ V -> [1,8,64,64]
    // CHECK: %[[RESULT:.*]] = "ttnn.matmul"(%[[SOFTMAX]], %arg2)
    // CHECK: return %[[RESULT]]

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }

  // Test 2: SDPA without scale attr (should use 1/sqrt(head_dim))
  // head_dim = 64, so scale = 1/sqrt(64) = 0.125
  func.func @sdpa_mha_default_scale(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_mha_default_scale

    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: %[[SCALE:.*]] = "ttnn.full"
    // CHECK-SAME: fill_value = 1.250000e-01
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir -v 2>&1 | tail -20
```

Expected: FAIL.

- [ ] **Step 3: Implement SDPA basic decomposition**

Replace the contents of
`lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include <cmath>

namespace mlir::tt::ttnn::decomposition {

// Dimension indices for [B, H, S, D] layout.
static constexpr int64_t kBatchDim = 0;
static constexpr int64_t kNumHeadsDim = 1;
static constexpr int64_t kSeqLenDim = 2;
static constexpr int64_t kHeadDim = 3;

LogicalResult SDPADecompositionPattern::matchAndRewrite(
    ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const {

  if (!forceDecompose) {
    FusionValidator validator(rewriter.getContext(), *validationConfig);

    auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());
    auto validationResult =
        validator.validateFusion<ScaledDotProductAttentionOp>(
            op.getOperation(), op.getLoc(), {qType}, op.getQuery(), op.getKey(),
            op.getValue(), op.getAttentionMask(), op.getIsCausal(),
            op.getScaleAttr(), op.getSlidingWindowSizeAttr(),
            op.getAttentionSink(),
            op.getMemoryConfigAttr());

    if (validationResult.isSuccess()) {
      return failure(); // Op is valid on device, no decomposition needed.
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                 "SDPA validation failed, decomposing: {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();
  auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());
  auto kType = mlir::cast<RankedTensorType>(op.getKey().getType());
  auto qShape = qType.getShape();
  auto kShape = kType.getShape();

  int64_t numHeads = qShape[kNumHeadsDim];
  int64_t numKVHeads = kShape[kNumHeadsDim];
  int64_t headDim = qShape[kHeadDim];

  Value key = op.getKey();
  Value value = op.getValue();

  // Step 1: GQA head expansion via repeat_interleave.
  if (numHeads != numKVHeads) {
    uint32_t repeats = static_cast<uint32_t>(numHeads / numKVHeads);

    llvm::SmallVector<int64_t> expandedKShape(kShape);
    expandedKShape[kNumHeadsDim] = numHeads;
    auto expandedKType =
        ttnn::utils::RankedTensorTypeFactory::create(kType, expandedKShape);

    key = rewriter.create<RepeatInterleaveOp>(
        loc, expandedKType, key, rewriter.getUI32IntegerAttr(repeats),
        rewriter.getSI32IntegerAttr(kNumHeadsDim),
        /*memory_config=*/MemoryConfigAttr());

    auto vType = mlir::cast<RankedTensorType>(value.getType());
    llvm::SmallVector<int64_t> expandedVShape(vType.getShape());
    expandedVShape[kNumHeadsDim] = numHeads;
    auto expandedVType =
        ttnn::utils::RankedTensorTypeFactory::create(vType, expandedVShape);

    value = rewriter.create<RepeatInterleaveOp>(
        loc, expandedVType, value, rewriter.getUI32IntegerAttr(repeats),
        rewriter.getSI32IntegerAttr(kNumHeadsDim),
        /*memory_config=*/MemoryConfigAttr());
  }

  // Step 2: Transpose K: [B, H, Skv, D] -> [B, H, D, Skv].
  auto curKType = mlir::cast<RankedTensorType>(key.getType());
  auto curKShape = curKType.getShape();
  llvm::SmallVector<int64_t> transposedKShape = {
      curKShape[0], curKShape[1], curKShape[3], curKShape[2]};
  auto transposedKType =
      ttnn::utils::RankedTensorTypeFactory::create(curKType, transposedKShape);

  Value keyTransposed = rewriter.create<TransposeOp>(
      loc, transposedKType, key,
      rewriter.getSI32IntegerAttr(-2), rewriter.getSI32IntegerAttr(-1));

  // Step 3: Matmul Q @ K^T -> scores [B, H, Sq, Skv].
  int64_t kvSeqLen = curKShape[kSeqLenDim];
  llvm::SmallVector<int64_t> scoresShape = {
      qShape[kBatchDim], numHeads, qShape[kSeqLenDim], kvSeqLen};
  auto scoresType =
      ttnn::utils::RankedTensorTypeFactory::create(qType, scoresShape);

  Value scores = rewriter.create<MatmulOp>(
      loc, scoresType, op.getQuery(), keyTransposed,
      /*transpose_a=*/rewriter.getBoolAttr(false),
      /*transpose_b=*/rewriter.getBoolAttr(false),
      /*matmul_program_config=*/nullptr,
      /*activation=*/nullptr,
      /*compute_config=*/nullptr);

  // Step 4: Scale (always explicit).
  float scaleValue;
  if (op.getScaleAttr()) {
    scaleValue = op.getScaleAttr().getValueAsDouble();
  } else {
    scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
  }

  auto fullOp = rewriter.create<FullOp>(
      loc, scoresType,
      /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(),
                           llvm::SmallVector<int64_t>(scoresShape)),
      rewriter.getF32FloatAttr(scaleValue),
      /*dtype=*/nullptr,
      /*layout=*/nullptr,
      /*memory_config=*/nullptr);
  scores = rewriter.create<MultiplyOp>(loc, scoresType, scores,
                                       fullOp.getResult());

  // Step 5: Add attention mask.
  if (op.getAttentionMask()) {
    scores = rewriter.create<AddOp>(loc, scoresType, scores,
                                    op.getAttentionMask());
  }

  // Step 6: Attention sink — concat on last dim before softmax.
  bool hasAttentionSink = op.getAttentionSink() != nullptr;
  if (hasAttentionSink) {
    auto sinkType =
        mlir::cast<RankedTensorType>(op.getAttentionSink().getType());
    int64_t sinkWidth = sinkType.getShape().back();

    llvm::SmallVector<int64_t> paddedScoresShape(scoresShape);
    paddedScoresShape.back() += sinkWidth;
    auto paddedScoresType =
        ttnn::utils::RankedTensorTypeFactory::create(qType, paddedScoresShape);

    SmallVector<Value> concatInputs = {scores, op.getAttentionSink()};
    scores = rewriter.create<ConcatOp>(
        loc, paddedScoresType, concatInputs,
        static_cast<int32_t>(scoresShape.size() - 1),
        /*memory_config=*/MemoryConfigAttr());
  }

  // Step 7: Softmax on last dimension.
  auto softmaxInputType = mlir::cast<RankedTensorType>(scores.getType());
  Value softmaxOut = rewriter.create<SoftmaxOp>(
      loc, softmaxInputType, scores,
      rewriter.getSI32IntegerAttr(-1),
      /*numericStable=*/rewriter.getBoolAttr(true),
      /*compute_config=*/nullptr);

  // Step 8: Slice to remove sink columns if attention_sink was present.
  if (hasAttentionSink) {
    int64_t rank = scoresShape.size();
    llvm::SmallVector<int32_t> begins(rank, 0);
    llvm::SmallVector<int32_t> ends;
    llvm::SmallVector<int32_t> steps(rank, 1);
    for (int64_t dim : scoresShape) {
      ends.push_back(static_cast<int32_t>(dim));
    }

    auto slicedType =
        ttnn::utils::RankedTensorTypeFactory::create(qType, scoresShape);
    softmaxOut = rewriter.create<SliceStaticOp>(
        loc, slicedType, softmaxOut,
        rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
        rewriter.getI32ArrayAttr(steps));
  }

  // Step 9: Matmul softmax @ V -> result [B, H, Sq, D].
  Value result = rewriter.create<MatmulOp>(
      loc, qType, softmaxOut, value,
      /*transpose_a=*/rewriter.getBoolAttr(false),
      /*transpose_b=*/rewriter.getBoolAttr(false),
      /*matmul_program_config=*/nullptr,
      /*activation=*/nullptr,
      /*compute_config=*/nullptr);

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
```

- [ ] **Step 4: Build and run test**

```bash
cmake --build build -- -j$(nproc) ttmlir-opt 2>&1 | tail -5
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Implement SDPA basic decomposition (MHA with mask)

Decomposes ttnn.scaled_dot_product_attention into: transpose K,
matmul Q@K^T, scale, add mask, softmax, matmul scores@V.
Handles GQA via repeat_interleave and attention_sink via
concat before softmax / slice after."
```

---

### Task 4: SDPA GQA and Attention Sink Tests

**Files:**
- Modify: `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`

- [ ] **Step 1: Add GQA and attention sink test cases**

Append to `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`:

```mlir
  // Test 3: GQA — 32 query heads, 8 KV heads (4:1 ratio)
  func.func @sdpa_gqa(
    %query: tensor<1x32x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x32x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_gqa

    // GQA: repeat_interleave K and V on dim 1 with repeats=4
    // CHECK: %[[K_EXP:.*]] = "ttnn.repeat_interleave"(%arg1)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32
    // CHECK: %[[V_EXP:.*]] = "ttnn.repeat_interleave"(%arg2)
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: repeats = 4 : ui32

    // CHECK: "ttnn.transpose"(%[[K_EXP]])
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.full"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x32x64x64xbf16>
    return %result : tensor<1x32x64x64xbf16>
  }

  // Test 4: MQA — 32 query heads, 1 KV head
  func.func @sdpa_mqa(
    %query: tensor<1x32x64x64xbf16>,
    %key: tensor<1x1x64x64xbf16>,
    %value: tensor<1x1x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>
  ) -> tensor<1x32x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_mqa
    // CHECK: "ttnn.repeat_interleave"(%arg1)
    // CHECK-SAME: repeats = 32 : ui32
    // CHECK: "ttnn.repeat_interleave"(%arg2)
    // CHECK-SAME: repeats = 32 : ui32
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x64x64xbf16>, tensor<1x1x64x64xbf16>,
         tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
      -> tensor<1x32x64x64xbf16>
    return %result : tensor<1x32x64x64xbf16>
  }

  // Test 5: SDPA with attention_sink
  func.func @sdpa_with_attention_sink(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>,
    %mask: tensor<1x1x64x64xbf16>,
    %sink: tensor<1x8x64x1xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_with_attention_sink

    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"

    // Concat scores with sink on last dim: [1,8,64,64] + [1,8,64,1] -> [1,8,64,65]
    // CHECK: %[[CONCAT:.*]] = "ttnn.concat"
    // CHECK-SAME: dim = 3

    // Softmax on padded scores
    // CHECK: %[[SOFTMAX:.*]] = "ttnn.softmax"(%[[CONCAT]])

    // Slice to remove sink column: [1,8,64,65] -> [1,8,64,64]
    // CHECK: %[[SLICED:.*]] = "ttnn.slice_static"(%[[SOFTMAX]])
    // CHECK-SAME: ends = [1 : i32, 8 : i32, 64 : i32, 64 : i32]

    // Final matmul
    // CHECK: "ttnn.matmul"(%[[SLICED]]

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask, %sink) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>,
         tensor<1x8x64x1xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }
```

- [ ] **Step 2: Run all tests**

```bash
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/ -v
```

Expected: All tests PASS (GQA and attention sink logic is already in Task 3 implementation).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add GQA, MQA, and attention sink decomposition tests"
```

---

### Task 5: Causal Mask Generation

**Files:**
- Modify: `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp`
- Modify: `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`

**References:**
- ArangeOp: `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` (search for `ArangeOp`)
- Existing arange usage: `test/unittests/OpModel/TTNN/Op/TestOpModelInterface.cpp:4880`

- [ ] **Step 1: Add causal mask test case**

Append to `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`:

```mlir
  // Test 6: Causal SDPA — is_causal=true, no explicit mask
  // Should generate causal mask via arange + comparison + where
  func.func @sdpa_causal_no_mask(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_causal_no_mask

    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"

    // Causal mask generation: arange ops + comparison + where
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.arange"

    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }
```

- [ ] **Step 2: Implement causal mask generation**

Add a helper function in `SDPADecompositionPattern.cpp` before `matchAndRewrite`:

```cpp
/// Generate an additive causal mask of shape [1, 1, Sq, Skv].
/// Lower triangle = 0.0, upper triangle = -inf.
/// Uses arange to build row/col indices, then compares and selects.
static Value generateCausalMask(PatternRewriter &rewriter, Location loc,
                                int64_t querySeqLen, int64_t keySeqLen,
                                RankedTensorType referenceType) {
  // Row indices: arange(0, Sq) -> [1, 1, Sq, 1], broadcast over Skv.
  // Col indices: arange(0, Skv) -> [1, 1, 1, Skv], broadcast over Sq.
  // Mask: where(row >= col, 0.0, -inf)
  //
  // Build row index tensor [1, 1, querySeqLen, 1].
  auto rowShape = llvm::SmallVector<int64_t>{1, 1, querySeqLen, 1};
  auto f32Type = rewriter.getF32Type();
  auto rowTensorType = RankedTensorType::get(rowShape, f32Type);
  Value rowIndices = rewriter.create<ArangeOp>(
      loc, rowTensorType,
      /*device=*/Value(),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(querySeqLen),
      rewriter.getI64IntegerAttr(1),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Build col index tensor [1, 1, 1, keySeqLen].
  auto colShape = llvm::SmallVector<int64_t>{1, 1, 1, keySeqLen};
  auto colTensorType = RankedTensorType::get(colShape, f32Type);
  Value colIndices = rewriter.create<ArangeOp>(
      loc, colTensorType,
      /*device=*/Value(),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(keySeqLen),
      rewriter.getI64IntegerAttr(1),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // Compare: row >= col -> boolean mask [1, 1, Sq, Skv] (via broadcast).
  auto maskShape = llvm::SmallVector<int64_t>{1, 1, querySeqLen, keySeqLen};
  auto boolTensorType = RankedTensorType::get(maskShape, rewriter.getI1Type());
  Value condition = rewriter.create<GreaterEqualOp>(
      loc, boolTensorType, rowIndices, colIndices);

  // Create 0.0 and -inf tensors.
  auto maskTensorType = RankedTensorType::get(maskShape, f32Type);
  Value zeros = rewriter.create<FullOp>(
      loc, maskTensorType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(0.0f),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  float negInf = -std::numeric_limits<float>::infinity();
  Value negInfTensor = rewriter.create<FullOp>(
      loc, maskTensorType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(negInf),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // where(condition, 0.0, -inf) -> additive causal mask.
  Value causalMask = rewriter.create<WhereOp>(
      loc, maskTensorType, condition, zeros, negInfTensor);

  return causalMask;
}
```

Then in `matchAndRewrite`, update Step 5 (mask handling) to call this:

```cpp
  // Step 5: Mask.
  if (op.getAttentionMask()) {
    scores = rewriter.create<AddOp>(loc, scoresType, scores,
                                    op.getAttentionMask());
  } else if (op.getIsCausal()) {
    Value causalMask = generateCausalMask(
        rewriter, loc, qShape[kSeqLenDim], kvSeqLen, qType);
    scores = rewriter.create<AddOp>(loc, scoresType, scores, causalMask);
  }
```

**Note:** You will need to verify the exact TTNN op names for comparison
(`GreaterEqualOp`) and conditional selection (`WhereOp`). Search for these
in `TTNNOps.td`. If these ops don't exist in TTNN, use the available
comparison/selection ops — search for `where`, `ge`, `greater_equal` in
the TTNN op definitions.

- [ ] **Step 3: Build and run tests**

```bash
cmake --build build -- -j$(nproc) ttmlir-opt 2>&1 | tail -5
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "Add causal mask generation for SDPA decomposition

When is_causal=true and no explicit mask is provided, generates
an additive causal mask using arange + comparison + where ops."
```

---

### Task 6: Sliding Window Mask Generation

**Files:**
- Modify: `lib/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.cpp`
- Modify: `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`

- [ ] **Step 1: Add sliding window test case**

Append to `test/ttmlir/Dialect/TTNN/decompose/sdpa_decompose.mlir`:

```mlir
  // Test 7: SDPA with sliding_window_size
  func.func @sdpa_sliding_window(
    %query: tensor<1x8x64x64xbf16>,
    %key: tensor<1x8x64x64xbf16>,
    %value: tensor<1x8x64x64xbf16>
  ) -> tensor<1x8x64x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_sliding_window
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"

    // Sliding window mask generation
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.arange"

    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
      is_causal = true,
      scale = 0.125 : f32,
      sliding_window_size = 32 : ui32
    }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
         tensor<1x8x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %result : tensor<1x8x64x64xbf16>
  }
```

- [ ] **Step 2: Implement sliding window mask generation**

Add a helper function in `SDPADecompositionPattern.cpp`:

```cpp
/// Generate sliding window mask of shape [1, 1, Sq, Skv].
/// Positions where (row - col) < 0 || (row - col) >= windowSize are -inf.
/// If combined with causal: intersect with lower-triangular constraint.
static Value generateSlidingWindowMask(PatternRewriter &rewriter, Location loc,
                                       int64_t querySeqLen, int64_t keySeqLen,
                                       uint32_t windowSize, bool isCausal,
                                       RankedTensorType referenceType) {
  auto f32Type = rewriter.getF32Type();
  auto maskShape = llvm::SmallVector<int64_t>{1, 1, querySeqLen, keySeqLen};

  // Row and col indices (same as causal mask).
  auto rowShape = llvm::SmallVector<int64_t>{1, 1, querySeqLen, 1};
  auto rowTensorType = RankedTensorType::get(rowShape, f32Type);
  Value rowIndices = rewriter.create<ArangeOp>(
      loc, rowTensorType, /*device=*/Value(),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(querySeqLen),
      rewriter.getI64IntegerAttr(1),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  auto colShape = llvm::SmallVector<int64_t>{1, 1, 1, keySeqLen};
  auto colTensorType = RankedTensorType::get(colShape, f32Type);
  Value colIndices = rewriter.create<ArangeOp>(
      loc, colTensorType, /*device=*/Value(),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(keySeqLen),
      rewriter.getI64IntegerAttr(1),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  // diff = row - col
  auto diffType = RankedTensorType::get(maskShape, f32Type);
  Value diff = rewriter.create<SubtractOp>(
      loc, diffType, rowIndices, colIndices);

  // Condition: diff >= 0 AND diff < windowSize
  // If causal: row >= col is already implied by diff >= 0.
  auto boolTensorType = RankedTensorType::get(maskShape, rewriter.getI1Type());

  // diff >= 0
  auto zeroTensor = rewriter.create<FullOp>(
      loc, diffType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(0.0f),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);
  Value geZero = rewriter.create<GreaterEqualOp>(
      loc, boolTensorType, diff, zeroTensor);

  // diff < windowSize
  auto windowTensor = rewriter.create<FullOp>(
      loc, diffType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(static_cast<float>(windowSize)),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);
  Value ltWindow = rewriter.create<LessThanOp>(
      loc, boolTensorType, diff, windowTensor);

  // Combined: geZero AND ltWindow
  Value condition = rewriter.create<LogicalAndOp>(
      loc, boolTensorType, geZero, ltWindow);

  // where(condition, 0.0, -inf)
  auto maskTensorType = RankedTensorType::get(maskShape, f32Type);
  Value zeros = rewriter.create<FullOp>(
      loc, maskTensorType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(0.0f),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  float negInf = -std::numeric_limits<float>::infinity();
  Value negInfTensor = rewriter.create<FullOp>(
      loc, maskTensorType, /*device=*/Value(),
      ttnn::ShapeAttr::get(rewriter.getContext(), maskShape),
      rewriter.getF32FloatAttr(negInf),
      /*dtype=*/nullptr, /*layout=*/nullptr, /*memory_config=*/nullptr);

  return rewriter.create<WhereOp>(
      loc, maskTensorType, condition, zeros, negInfTensor);
}
```

Update Step 5 in `matchAndRewrite` to handle sliding window:

```cpp
  // Step 5: Mask.
  if (op.getAttentionMask()) {
    scores = rewriter.create<AddOp>(loc, scoresType, scores,
                                    op.getAttentionMask());
  }

  if (op.getSlidingWindowSizeAttr()) {
    uint32_t windowSize = op.getSlidingWindowSizeAttr().getUInt();
    Value windowMask = generateSlidingWindowMask(
        rewriter, loc, qShape[kSeqLenDim], kvSeqLen,
        windowSize, op.getIsCausal(), qType);
    scores = rewriter.create<AddOp>(loc, scoresType, scores, windowMask);
  } else if (!op.getAttentionMask() && op.getIsCausal()) {
    Value causalMask = generateCausalMask(
        rewriter, loc, qShape[kSeqLenDim], kvSeqLen, qType);
    scores = rewriter.create<AddOp>(loc, scoresType, scores, causalMask);
  }
```

**Note:** The exact op names for `SubtractOp`, `LessThanOp`, `LogicalAndOp`
need to be verified against `TTNNOps.td`. Search for `subtract`, `less`,
`logical_and` to find the correct names.

- [ ] **Step 3: Build and run tests**

```bash
cmake --build build -- -j$(nproc) ttmlir-opt 2>&1 | tail -5
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "Add sliding window mask generation for SDPA decomposition

Generates combined causal + sliding window mask when
sliding_window_size attribute is present."
```

---

### Task 7: End-to-End Cascade Test (Decode -> SDPA -> Components)

**Files:**
- Create: `test/ttmlir/Dialect/TTNN/decompose/sdpa_cascade_decompose.mlir`

- [ ] **Step 1: Write cascade test**

Create `test/ttmlir/Dialect/TTNN/decompose/sdpa_cascade_decompose.mlir`:

```mlir
// RUN: ttmlir-opt --ttnn-decompose-ops-on-validation-failure="force-decompose=true" %s | FileCheck %s

module {
  // Test: SDPADecode should cascade through SDPA to component ops.
  // The greedy driver first decomposes decode->SDPA, then SDPA->components.
  func.func @sdpa_decode_full_cascade(
    %query: tensor<1x32x8x64xbf16>,
    %key: tensor<32x8x128x64xbf16>,
    %value: tensor<32x8x128x64xbf16>,
    %mask: tensor<32x1x1x128xbf16>
  ) -> tensor<1x32x8x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_decode_full_cascade

    // Should NOT see any SDPA ops — fully decomposed to components.
    // CHECK-NOT: ttnn.scaled_dot_product_attention_decode
    // CHECK-NOT: ttnn.scaled_dot_product_attention

    // Should see permute (from decode), then component ops, then permute back.
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.transpose"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.softmax"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.permute"

    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x8x64xbf16>, tensor<32x8x128x64xbf16>,
         tensor<32x8x128x64xbf16>, tensor<32x1x1x128xbf16>)
      -> tensor<1x32x8x64xbf16>
    return %result : tensor<1x32x8x64xbf16>
  }
}
```

- [ ] **Step 2: Run test**

```bash
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/sdpa_cascade_decompose.mlir -v
```

Expected: PASS — the greedy driver cascades both patterns.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "Add cascade test: SDPADecode decomposes fully to component ops"
```

---

### Task 8: Final Review and Cleanup

- [ ] **Step 1: Run all decomposition tests**

```bash
build/bin/llvm-lit test/ttmlir/Dialect/TTNN/decompose/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run pre-commit checks**

```bash
pre-commit run --all-files 2>&1 | tail -20
```

Fix any formatting issues.

- [ ] **Step 3: Run broader compiler tests to check for regressions**

```bash
cmake --build build --target check-ttmlir 2>&1 | tail -30
```

Expected: No regressions — the pass only runs with `forceDecompose=true` in
tests or when `enableOpConstraints=true` with optimizer, and validation-passing
ops are left untouched.

- [ ] **Step 4: Fix any issues found, commit if needed**

```bash
git add -A && git commit -m "Fix formatting and address review feedback"
```
