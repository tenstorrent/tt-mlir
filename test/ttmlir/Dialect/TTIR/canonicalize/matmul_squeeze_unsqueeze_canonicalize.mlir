// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Both inputs squeezed, output unsqueezed -- pattern fires.
  func.func @matmul_both_inputs_squeezed(%arg0: tensor<1x5x64x32xbf16>, %arg1: tensor<1x5x32x64xbf16>) -> tensor<1x5x64x64xbf16> {
    // CHECK-LABEL: @matmul_both_inputs_squeezed
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<1x5x64x64xbf16>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [5 : i32, 64 : i32, 32 : i32]}> : (tensor<1x5x64x32xbf16>) -> tensor<5x64x32xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [5 : i32, 32 : i32, 64 : i32]}> : (tensor<1x5x32x64xbf16>) -> tensor<5x32x64xbf16>
    %2 = "ttir.matmul"(%0, %1) : (tensor<5x64x32xbf16>, tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 5 : i32, 64 : i32, 64 : i32]}> : (tensor<5x64x64xbf16>) -> tensor<1x5x64x64xbf16>
    return %3 : tensor<1x5x64x64xbf16>
  }

  // Only input A squeezed -- pattern fires via batch broadcasting.
  func.func @matmul_one_input_squeezed(%arg0: tensor<1x64x128xbf16>, %arg1: tensor<128x32xbf16>) -> tensor<1x64x32xbf16> {
    // CHECK-LABEL: @matmul_one_input_squeezed
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<1x64x32xbf16>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32, 128 : i32]}> : (tensor<1x64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "ttir.matmul"(%0, %arg1) : (tensor<64x128xbf16>, tensor<128x32xbf16>) -> tensor<64x32xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 64 : i32, 32 : i32]}> : (tensor<64x32xbf16>) -> tensor<1x64x32xbf16>
    return %2 : tensor<1x64x32xbf16>
  }

  // No squeeze/unsqueeze reshapes -- pattern does not fire.
  func.func @matmul_no_reshape(%arg0: tensor<5x64x32xbf16>, %arg1: tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16> {
    // CHECK-LABEL: @matmul_no_reshape
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<5x64x64xbf16>
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<5x64x32xbf16>, tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16>
    return %0 : tensor<5x64x64xbf16>
  }

  // Matmul result has multiple uses -- pattern does not fire.
  func.func @matmul_multi_use(%arg0: tensor<1x5x64x32xbf16>, %arg1: tensor<1x5x32x64xbf16>) -> (tensor<1x5x64x64xbf16>, tensor<5x64x64xbf16>) {
    // CHECK-LABEL: @matmul_multi_use
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [5 : i32, 64 : i32, 32 : i32]}> : (tensor<1x5x64x32xbf16>) -> tensor<5x64x32xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [5 : i32, 32 : i32, 64 : i32]}> : (tensor<1x5x32x64xbf16>) -> tensor<5x32x64xbf16>
    %2 = "ttir.matmul"(%0, %1) : (tensor<5x64x32xbf16>, tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 5 : i32, 64 : i32, 64 : i32]}> : (tensor<5x64x64xbf16>) -> tensor<1x5x64x64xbf16>
    return %3, %2 : tensor<1x5x64x64xbf16>, tensor<5x64x64xbf16>
  }

  // Transpose attributes are preserved -- pattern fires.
  func.func @matmul_with_transpose(%arg0: tensor<1x5x32x64xbf16>, %arg1: tensor<1x5x64x32xbf16>) -> tensor<1x5x64x64xbf16> {
    // CHECK-LABEL: @matmul_with_transpose
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: -> tensor<1x5x64x64xbf16>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [5 : i32, 32 : i32, 64 : i32]}> : (tensor<1x5x32x64xbf16>) -> tensor<5x32x64xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [5 : i32, 64 : i32, 32 : i32]}> : (tensor<1x5x64x32xbf16>) -> tensor<5x64x32xbf16>
    %2 = "ttir.matmul"(%0, %1) <{transpose_a = true, transpose_b = true}> : (tensor<5x32x64xbf16>, tensor<5x64x32xbf16>) -> tensor<5x64x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 5 : i32, 64 : i32, 64 : i32]}> : (tensor<5x64x64xbf16>) -> tensor<1x5x64x64xbf16>
    return %3 : tensor<1x5x64x64xbf16>
  }

  // Non-leading squeeze (middle dim) -- pattern does not fire.
  func.func @matmul_non_leading_squeeze(%arg0: tensor<5x1x64x32xbf16>, %arg1: tensor<5x32x64xbf16>) -> tensor<5x1x64x64xbf16> {
    // CHECK-LABEL: @matmul_non_leading_squeeze
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [5 : i32, 64 : i32, 32 : i32]}> : (tensor<5x1x64x32xbf16>) -> tensor<5x64x32xbf16>
    %1 = "ttir.matmul"(%0, %arg1) : (tensor<5x64x32xbf16>, tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [5 : i32, 1 : i32, 64 : i32, 64 : i32]}> : (tensor<5x64x64xbf16>) -> tensor<5x1x64x64xbf16>
    return %2 : tensor<5x1x64x64xbf16>
  }

  // Only A squeezed (rank 4->3), B is batched rank-3 -- pattern does NOT fire.
  // Both new inputs would be rank >= 3 with different ranks; this would cause
  // TTNN BMM to fail with "a_shape.rank() == b_shape.rank()".
  func.func @matmul_only_a_squeezed_b_batched(%arg0: tensor<1x40x6240x128xbf16>, %arg1: tensor<40x128x49920xbf16>) -> tensor<1x40x6240x49920xbf16> {
    // CHECK-LABEL: @matmul_only_a_squeezed_b_batched
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK-SAME: tensor<40x6240x128xbf16>, tensor<40x128x49920xbf16>
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [40 : i32, 6240 : i32, 128 : i32]}> : (tensor<1x40x6240x128xbf16>) -> tensor<40x6240x128xbf16>
    %1 = "ttir.matmul"(%0, %arg1) : (tensor<40x6240x128xbf16>, tensor<40x128x49920xbf16>) -> tensor<40x6240x49920xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 40 : i32, 6240 : i32, 49920 : i32]}> : (tensor<40x6240x49920xbf16>) -> tensor<1x40x6240x49920xbf16>
    return %2 : tensor<1x40x6240x49920xbf16>
  }

  // Multiple leading 1s squeezed -- pattern fires.
  func.func @matmul_multiple_leading_ones(%arg0: tensor<1x1x5x64x32xbf16>, %arg1: tensor<1x1x5x32x64xbf16>) -> tensor<1x1x5x64x64xbf16> {
    // CHECK-LABEL: @matmul_multiple_leading_ones
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<1x1x5x64x64xbf16>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [5 : i32, 64 : i32, 32 : i32]}> : (tensor<1x1x5x64x32xbf16>) -> tensor<5x64x32xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [5 : i32, 32 : i32, 64 : i32]}> : (tensor<1x1x5x32x64xbf16>) -> tensor<5x32x64xbf16>
    %2 = "ttir.matmul"(%0, %1) : (tensor<5x64x32xbf16>, tensor<5x32x64xbf16>) -> tensor<5x64x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 5 : i32, 64 : i32, 64 : i32]}> : (tensor<5x64x64xbf16>) -> tensor<1x1x5x64x64xbf16>
    return %3 : tensor<1x1x5x64x64xbf16>
  }
}
