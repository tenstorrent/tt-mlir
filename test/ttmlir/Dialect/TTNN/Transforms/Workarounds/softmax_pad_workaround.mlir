// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

module @test_softmax_pad_workaround attributes {} {
  // Test 1: Workaround SHOULD apply - softmax on last dimension
  func.func public @test_softmax_pad_workaround_last_dim(
    %input: tensor<1x1x6240x6240xf32>
  ) -> tensor<1x1x6240x6240xf32> {
    // CHECK-LABEL: func.func public @test_softmax_pad_workaround_last_dim

    // CHECK: %[[PADDED_INPUT:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 32>

    // CHECK: %[[SOFTMAX:[0-9]+]] = "ttnn.softmax"(%[[PADDED_INPUT]])
    // CHECK-SAME: dimension = 3

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SOFTMAX]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 1 : i32, 6240 : i32, 6240 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]

    // CHECK: return %[[SLICED]]
    %result = "ttnn.softmax"(%input) {dimension = 3: si32} : (tensor<1x1x6240x6240xf32>) -> tensor<1x1x6240x6240xf32>
    return %result : tensor<1x1x6240x6240xf32>
  }
  // Test 2: Workaround should NOT apply - tensor is not rank-4 (rank-3 instead)
  func.func public @test_softmax_pad_workaround_non_rank4(
    %input: tensor<1x6240x6240xf32>
  ) -> tensor<1x6240x6240xf32> {
    // CHECK-LABEL: func.func public @test_softmax_pad_workaround_non_rank4
    // CHECK: "ttnn.softmax"(
    // CHECK-SAME: dimension = 2
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %result = "ttnn.softmax"(%input) {dimension = 2: si32} : (tensor<1x6240x6240xf32>) -> tensor<1x6240x6240xf32>
    return %result : tensor<1x6240x6240xf32>
  }
  // Test 3: Workaround should NOT apply - softmax is not on the last dimension
  func.func public @test_softmax_pad_workaround_not_last_dim(
    %input: tensor<1x1x6240x6240xf32>
  ) -> tensor<1x1x6240x6240xf32> {
    // CHECK-LABEL: func.func public @test_softmax_pad_workaround_not_last_dim
    // CHECK: "ttnn.softmax"(
    // CHECK-SAME: dimension = 2
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %result = "ttnn.softmax"(%input) {dimension = 2: si32} : (tensor<1x1x6240x6240xf32>) -> tensor<1x1x6240x6240xf32>
    return %result : tensor<1x1x6240x6240xf32>
  }
  // Test 4: Workaround should NOT apply - width already produces a safe block size
  // width = 6272 (196 tiles) => block_size = 4, which divides 80
  func.func public @test_softmax_pad_workaround_safe_width(
    %input: tensor<1x1x6272x6272xf32>
  ) -> tensor<1x1x6272x6272xf32> {
    // CHECK-LABEL: func.func public @test_softmax_pad_workaround_safe_width
    // CHECK: "ttnn.softmax"(
    // CHECK-SAME: dimension = 3
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %result = "ttnn.softmax"(%input) {dimension = 3: si32} : (tensor<1x1x6272x6272xf32>) -> tensor<1x1x6272x6272xf32>
    return %result : tensor<1x1x6272x6272xf32>
  }
}
