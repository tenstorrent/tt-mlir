// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // Both inputs merged, output split -- pattern fires.
  // This is the SDPA pattern: [batch, heads, S, D] -> [batch*heads, S, D]
  func.func @matmul_both_merged(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<32x32x64x128xf32>) -> tensor<32x32x18x128xf32> {
    // CHECK-LABEL: @matmul_both_merged
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<32x32x18x128xf32>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 64 : i32, 128 : i32]}> : (tensor<32x32x64x128xf32>) -> tensor<1024x64x128xf32>
    %2 = "ttir.matmul"(%0, %1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<32x32x18x128xf32>
    return %3 : tensor<32x32x18x128xf32>
  }

  // Transpose attributes are preserved -- pattern fires.
  func.func @matmul_merged_with_transpose(%arg0: tensor<32x32x64x18xf32>, %arg1: tensor<32x32x128x64xf32>) -> tensor<32x32x18x128xf32> {
    // CHECK-LABEL: @matmul_merged_with_transpose
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = true
    // CHECK-SAME: -> tensor<32x32x18x128xf32>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 64 : i32, 18 : i32]}> : (tensor<32x32x64x18xf32>) -> tensor<1024x64x18xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 128 : i32, 64 : i32]}> : (tensor<32x32x128x64xf32>) -> tensor<1024x128x64xf32>
    %2 = "ttir.matmul"(%0, %1) <{transpose_a = true, transpose_b = true}> : (tensor<1024x64x18xf32>, tensor<1024x128x64xf32>) -> tensor<1024x18x128xf32>
    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<32x32x18x128xf32>
    return %3 : tensor<32x32x18x128xf32>
  }

  // Three leading dims merged -- pattern fires.
  func.func @matmul_three_dims_merged(%arg0: tensor<4x8x16x64x32xbf16>, %arg1: tensor<4x8x16x32x64xbf16>) -> tensor<4x8x16x64x64xbf16> {
    // CHECK-LABEL: @matmul_three_dims_merged
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.matmul"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<4x8x16x64x64xbf16>
    // CHECK-NOT: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [512 : i32, 64 : i32, 32 : i32]}> : (tensor<4x8x16x64x32xbf16>) -> tensor<512x64x32xbf16>
    %1 = "ttir.reshape"(%arg1) <{shape = [512 : i32, 32 : i32, 64 : i32]}> : (tensor<4x8x16x32x64xbf16>) -> tensor<512x32x64xbf16>
    %2 = "ttir.matmul"(%0, %1) : (tensor<512x64x32xbf16>, tensor<512x32x64xbf16>) -> tensor<512x64x64xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [4 : i32, 8 : i32, 16 : i32, 64 : i32, 64 : i32]}> : (tensor<512x64x64xbf16>) -> tensor<4x8x16x64x64xbf16>
    return %3 : tensor<4x8x16x64x64xbf16>
  }

  // Only one input merged -- pattern does not fire.
  func.func @matmul_one_input_merged(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<1024x64x128xf32>) -> tensor<32x32x18x128xf32> {
    // CHECK-LABEL: @matmul_one_input_merged
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.matmul"(%0, %arg1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<32x32x18x128xf32>
    return %2 : tensor<32x32x18x128xf32>
  }

  // Leading dims don't match between A and B -- pattern does not fire.
  func.func @matmul_mismatched_leading_dims(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<64x16x64x128xf32>) -> tensor<32x32x18x128xf32> {
    // CHECK-LABEL: @matmul_mismatched_leading_dims
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 64 : i32, 128 : i32]}> : (tensor<64x16x64x128xf32>) -> tensor<1024x64x128xf32>
    %2 = "ttir.matmul"(%0, %1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<32x32x18x128xf32>
    return %3 : tensor<32x32x18x128xf32>
  }

  // Split dims don't match merge dims -- pattern does not fire.
  func.func @matmul_mismatched_split(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<32x32x64x128xf32>) -> tensor<64x16x18x128xf32> {
    // CHECK-LABEL: @matmul_mismatched_split
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 64 : i32, 128 : i32]}> : (tensor<32x32x64x128xf32>) -> tensor<1024x64x128xf32>
    %2 = "ttir.matmul"(%0, %1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    %3 = "ttir.reshape"(%2) <{shape = [64 : i32, 16 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<64x16x18x128xf32>
    return %3 : tensor<64x16x18x128xf32>
  }

  // Matmul result has multiple uses -- pattern does not fire.
  func.func @matmul_merged_multi_use(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<32x32x64x128xf32>) -> (tensor<32x32x18x128xf32>, tensor<1024x18x128xf32>) {
    // CHECK-LABEL: @matmul_merged_multi_use
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 64 : i32, 128 : i32]}> : (tensor<32x32x64x128xf32>) -> tensor<1024x64x128xf32>
    %2 = "ttir.matmul"(%0, %1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<1024x18x128xf32>) -> tensor<32x32x18x128xf32>
    return %3, %2 : tensor<32x32x18x128xf32>, tensor<1024x18x128xf32>
  }

  // No output reshape -- pattern does not fire.
  func.func @matmul_merged_no_split(%arg0: tensor<32x32x18x64xf32>, %arg1: tensor<32x32x64x128xf32>) -> tensor<1024x18x128xf32> {
    // CHECK-LABEL: @matmul_merged_no_split
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.matmul"
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 18 : i32, 64 : i32]}> : (tensor<32x32x18x64xf32>) -> tensor<1024x18x64xf32>
    %1 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 64 : i32, 128 : i32]}> : (tensor<32x32x64x128xf32>) -> tensor<1024x64x128xf32>
    %2 = "ttir.matmul"(%0, %1) : (tensor<1024x18x64xf32>, tensor<1024x64x128xf32>) -> tensor<1024x18x128xf32>
    return %2 : tensor<1024x18x128xf32>
  }
}
