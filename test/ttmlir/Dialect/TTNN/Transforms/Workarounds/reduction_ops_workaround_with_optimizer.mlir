// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression test: the reduction ops (sum/mean/max/min) must remain in the
// workaround whitelist when the optimizer is enabled.
//
// The TTNN workarounds pass restricts itself to a small set of ops when
// `optimizer-enabled=true` (opt_level >= 1). The reduction ops were previously
// absent from that set, so `createReductionOpOperandsWorkarounds`
// (TTNNWorkaroundsPass.cpp) was skipped at opt >= 1. Without that workaround
// firing, an integer-typed reduction input is left in-place and the optimizer's
// dtype propagation routes it through BFloat16 rather than Float32. BFloat16 has
// only 8 mantissa bits, so integer values above 2^8 are silently rounded (e.g.
// 8400 -> 8384), corrupting integer reductions such as the cross-entropy / NLL
// loss sum-reductions over vocab-sized index tensors.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/21071
// tt-mlir issue:  https://github.com/tenstorrent/tt-mlir/issues/8279
//
// These tests assert the same behavior as `reduction_ops_workaround.mlir`
// (integer input -> Float32 reduction -> to_layout back to the originally
// requested si32) but with the optimizer-enabled path engaged. Without the
// reduction ops in `enabledOpsForWorkaroundWithOptimizer`, the to_layout to f32
// and back to si32 below would be absent and these tests would fail.

module {
  func.func public @test_reduce_sum_workaround_with_optimizer(%arg0: tensor<128x10xsi32>) -> tensor<128xsi32> {
    // CHECK-LABEL: func.func public @test_reduce_sum_workaround_with_optimizer
    // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: tensor<128x10xsi32,
    // CHECK-SAME: -> tensor<128x10xf32,
    // CHECK: %[[SUM:.*]] = "ttnn.sum"(%[[ARG0]])
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xsi32>) -> tensor<128xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[SUM]])
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<128xsi32,
    return %0 : tensor<128xsi32>
  }
}

// -----
module {
  func.func public @test_reduce_mean_workaround_with_optimizer(%arg0: tensor<128x10xsi32>) -> tensor<128xsi32> {
    // CHECK-LABEL: func.func public @test_reduce_mean_workaround_with_optimizer
    // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: tensor<128x10xsi32,
    // CHECK-SAME: -> tensor<128x10xf32,
    // CHECK: %[[MEAN:.*]] = "ttnn.mean"(%[[ARG0]])
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10xsi32>) -> tensor<128xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MEAN]])
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<128xsi32,
    return %0 : tensor<128xsi32>
  }
}

// -----
module {
  func.func public @test_reduce_max_workaround_with_optimizer(%arg0: tensor<128x32xsi32>) -> tensor<128xsi32> {
    // CHECK-LABEL: func.func public @test_reduce_max_workaround_with_optimizer
    // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: tensor<128x32xsi32,
    // CHECK-SAME: -> tensor<128x32xf32,
    // CHECK: %[[MAX:.*]] = "ttnn.max"(%[[ARG0]])
    // CHECK-SAME: tensor<128x32xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x32xsi32>) -> tensor<128xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MAX]])
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<128xsi32,
    return %0 : tensor<128xsi32>
  }
}

// -----
module {
  func.func public @test_reduce_min_workaround_with_optimizer(%arg0: tensor<128x32xsi32>) -> tensor<128xsi32> {
    // CHECK-LABEL: func.func public @test_reduce_min_workaround_with_optimizer
    // CHECK: %[[ARG0:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: tensor<128x32xsi32,
    // CHECK-SAME: -> tensor<128x32xf32,
    // CHECK: %[[MIN:.*]] = "ttnn.min"(%[[ARG0]])
    // CHECK-SAME: tensor<128x32xf32,
    // CHECK-SAME: -> tensor<128xf32,
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x32xsi32>) -> tensor<128xsi32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[MIN]])
    // CHECK-SAME: tensor<128xf32,
    // CHECK-SAME: -> tensor<128xsi32,
    return %0 : tensor<128xsi32>
  }
}
