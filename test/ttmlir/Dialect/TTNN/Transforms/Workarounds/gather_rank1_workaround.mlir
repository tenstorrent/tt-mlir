// RUN: ttmlir-opt -ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// tt-metal's gather requires rank >= 2 operands. The rank-1 workaround
// unsqueezes a leading unit dimension on the input and index, gathers on the
// resulting rank-2 tensors, and reshapes the result back to rank 1.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/45155

module {
  // Test: rank-1 input/index with dim 0 -- unsqueezed to rank 2 and gathered.
  // dim 0 maps to dim 1 under the leading unit-dim prepend.
  func.func public @gather_rank1(%arg0: tensor<5xf32>, %arg1: tensor<2xui32>) -> tensor<2xf32> {
    // CHECK-LABEL: func.func public @gather_rank1
    // CHECK: "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 5 : i32]
    // CHECK-SAME: -> tensor<1x5xf32
    // CHECK: "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 2 : i32]
    // CHECK-SAME: -> tensor<1x2xui32
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = 1 : i32
    // CHECK-SAME: (tensor<1x5xf32, {{.*}}>, tensor<1x2xui32, {{.*}}>) -> tensor<1x2xf32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [2 : i32]
    // CHECK-SAME: -> tensor<2xf32
    %0 = "ttnn.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5xf32>, tensor<2xui32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // Test: rank-1 input/index with negative dim -- the dim already counts from
  // the back and is passed through to the rank-2 gather unchanged.
  func.func public @gather_rank1_negative_dim(%arg0: tensor<5xf32>, %arg1: tensor<2xui32>) -> tensor<2xf32> {
    // CHECK-LABEL: func.func public @gather_rank1_negative_dim
    // CHECK: "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 5 : i32]
    // CHECK-SAME: -> tensor<1x5xf32
    // CHECK: "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 2 : i32]
    // CHECK-SAME: -> tensor<1x2xui32
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = -1 : i32
    // CHECK-SAME: (tensor<1x5xf32, {{.*}}>, tensor<1x2xui32, {{.*}}>) -> tensor<1x2xf32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [2 : i32]
    // CHECK-SAME: -> tensor<2xf32
    %0 = "ttnn.gather"(%arg0, %arg1) <{dim = -1 : i32}> : (tensor<5xf32>, tensor<2xui32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // Test: rank-2 input/index -- already rank >= 2, workaround should not fire.
  func.func public @gather_rank2_noop(%arg0: tensor<5x3xf32>, %arg1: tensor<2x3xui32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func public @gather_rank2_noop
    // CHECK-NOT: "ttnn.reshape"
    // CHECK: "ttnn.gather"
    // CHECK-SAME: dim = 0 : i32
    %0 = "ttnn.gather"(%arg0, %arg1) <{dim = 0 : i32}> : (tensor<5x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
