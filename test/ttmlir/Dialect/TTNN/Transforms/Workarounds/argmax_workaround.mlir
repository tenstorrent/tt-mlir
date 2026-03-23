// RUN: ttmlir-opt -ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test: dim=0 on 3D tensor should be permuted so argmax runs on last dim.
  func.func public @argmax_3d_dim0(%arg0: tensor<128x32x64xf32>) -> tensor<32x64xui32> {
    // CHECK-LABEL: func.func public @argmax_3d_dim0
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: dim = 2 : i32
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0>
    %0 = "ttnn.argmax"(%arg0) <{dim = 0 : i32, keep_dim = false, use_multicore = false}> : (tensor<128x32x64xf32>) -> tensor<32x64xui32>
    return %0 : tensor<32x64xui32>
  }

  // Test: dim=1 on 4D tensor with keep_dim=true.
  func.func public @argmax_4d_dim1_keepdim(%arg0: tensor<2x8x32x64xf32>) -> tensor<2x1x32x64xui32> {
    // CHECK-LABEL: func.func public @argmax_4d_dim1_keepdim
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 2, 1>
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: dim = 3 : i32
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 2, 1>
    %0 = "ttnn.argmax"(%arg0) <{dim = 1 : i32, keep_dim = true, use_multicore = false}> : (tensor<2x8x32x64xf32>) -> tensor<2x1x32x64xui32>
    return %0 : tensor<2x1x32x64xui32>
  }

  // Test: dim is already last dim, workaround should not trigger.
  func.func public @argmax_last_dim(%arg0: tensor<32x64xf32>) -> tensor<32xui32> {
    // CHECK-LABEL: func.func public @argmax_last_dim
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.argmax"
    // CHECK-SAME: dim = 1 : i32
    %0 = "ttnn.argmax"(%arg0) <{dim = 1 : i32, keep_dim = false, use_multicore = false}> : (tensor<32x64xf32>) -> tensor<32xui32>
    return %0 : tensor<32xui32>
  }
}
