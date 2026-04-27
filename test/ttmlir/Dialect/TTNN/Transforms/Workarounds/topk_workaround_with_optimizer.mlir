// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround="ttnn-is-optimizer-enabled=true" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression test: TopKOp must remain in the workaround whitelist when the
// optimizer is enabled.
//
// The TTNN workarounds pass restricts itself to a small set of ops when
// `optimizer-enabled=true` (chain-based or greedy optimizer paths at
// opt_level >= 1). TopKOp was previously absent from that set, so its
// `createTopKOpOperandsWorkarounds` (TTNNWorkaroundsPass.cpp) was skipped
// at opt >= 1. Without that workaround firing, the optimizer's own dtype
// propagation filled in for the topk indices output, producing f32 typecasts
// instead of integer typecasts for some graph patterns (notably the
// chunked-topk pre-filter used by tt-xla's vLLM sampler, see
// integrations/vllm_plugin/vllm_tt/sampler.py::apply_top_k_top_p_fast).
// Downstream the chunk-offset add then had mismatched (f32, si32) operands
// with a declared si32 result, producing silently wrong global vocab
// indices. End-to-end symptom: garbage tokens during non-greedy device
// sampling on Llama at opt_level=1.
//
// These tests assert the same behavior as `topk_workaround.mlir` (input
// bf16/bfp_bf8 -> ui16/ui32 indices via metal kernel -> to_layout back to
// the originally-requested si32) but with the optimizer-enabled path
// engaged. Without TopKOp in `enabledOpsForWorkaroundWithOptimizer`, the
// to_layout to si32 below would be absent and these tests would fail.

module {
  func.func public @test_topk_workaround_with_optimizer_ui16(%arg0: tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>) {
    // CHECK-LABEL: func.func public @test_topk_workaround_with_optimizer_ui16
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttnn.topk"
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128xbf16,
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16,
    // CHECK-SAME: tensor<2x3x32x5xui16,
    %values, %indices = "ttir.topk"(%arg0) { k = 5 : i32 } : (tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>)
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x3x32x5xui16,
    // CHECK-SAME: -> tensor<2x3x32x5xsi32,
    return %values, %indices : tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>
  }
}

// -----
module {
  func.func public @test_topk_workaround_with_optimizer_ui32(%arg0: tensor<2x3x32x128000xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>) {
    // CHECK-LABEL: func.func public @test_topk_workaround_with_optimizer_ui32
    // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttnn.topk"
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128000xbf16,
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16,
    // CHECK-SAME: tensor<2x3x32x5xui32,
    %values, %indices = "ttir.topk"(%arg0) { k = 5 : i32 } : (tensor<2x3x32x128000xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>)
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x3x32x5xui32,
    // CHECK-SAME: -> tensor<2x3x32x5xsi32,
    return %values, %indices : tensor<2x3x32x5xbf16>, tensor<2x3x32x5xsi32>
  }
}

// -----
// Bonus case: f32 input must also be converted to bf16 by the workaround,
// matching `topk_f32_input_workaround.mlir` but with optimizer-enabled.

module {
  func.func public @test_topk_workaround_with_optimizer_f32_input(%arg0: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>) {
    // CHECK-LABEL: func.func public @test_topk_workaround_with_optimizer_f32_input
    // CHECK: %[[INPUT_BF16:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<2x3x32x128xf32
    // CHECK-SAME: -> tensor<2x3x32x128xbf16
    // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.topk"(%[[INPUT_BF16]])
    // CHECK-SAME: <{dim = -1 : i32, k = 5 : i32, largest = true, sorted = false}>
    // CHECK-SAME: tensor<2x3x32x128xbf16
    // CHECK-SAME: -> (tensor<2x3x32x5xbf16
    // CHECK-SAME: tensor<2x3x32x5xui16
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[INDICES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<2x3x32x5xui16
    // CHECK-SAME: -> tensor<2x3x32x5xsi32
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[VALUES]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<2x3x32x5xbf16
    // CHECK-SAME: -> tensor<2x3x32x5xf32
    %values, %indices = "ttir.topk"(%arg0) { k = 5 : i32 } : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>)
    return %values, %indices : tensor<2x3x32x5xf32>, tensor<2x3x32x5xsi32>
  }
}
