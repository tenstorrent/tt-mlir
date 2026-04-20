// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // LLaMA-style: 32x8x1x64 -> 1x32x8x64 (only singleton dim moves).
  func.func @permute_llama_like(%arg0: tensor<32x8x1x64xbf16>)
      -> tensor<1x32x8x64xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 2, 0, 1, 3> }
         : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    func.return %0 : tensor<1x32x8x64xbf16>
  }

  // CHECK-LABEL: @permute_llama_like(
  // CHECK: %[[R0:.*]] = "ttnn.reshape"(%arg0)
  // CHECK-SAME: shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]
  // CHECK-SAME: : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
  // CHECK: return %[[R0]] : tensor<1x32x8x64xbf16>

  // Same pattern, different batch size.
  func.func @permute_llama_like_other_batch(%arg0: tensor<16x8x1x64xbf16>)
      -> tensor<1x16x8x64xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 2, 0, 1, 3> }
         : (tensor<16x8x1x64xbf16>) -> tensor<1x16x8x64xbf16>
    func.return %0 : tensor<1x16x8x64xbf16>
  }

  // CHECK-LABEL: @permute_llama_like_other_batch(
  // CHECK: %[[R1:.*]] = "ttnn.reshape"(%arg0)
  // CHECK-SAME: shape = [1 : i32, 16 : i32, 8 : i32, 64 : i32]
  // CHECK-SAME: : (tensor<16x8x1x64xbf16>) -> tensor<1x16x8x64xbf16>
  // CHECK: return %[[R1]] : tensor<1x16x8x64xbf16>

  // Multiple singletons: only 1-dims move.
  func.func @permute_two_singletons(%arg0: tensor<2x1x3x1xbf16>)
      -> tensor<1x1x2x3xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 1, 3, 0, 2> }
         : (tensor<2x1x3x1xbf16>) -> tensor<1x1x2x3xbf16>
    func.return %0 : tensor<1x1x2x3xbf16>
  }

  // CHECK-LABEL: @permute_two_singletons(
  // CHECK: %[[R2:.*]] = "ttnn.reshape"(%arg0)
  // CHECK-SAME: shape = [1 : i32, 1 : i32, 2 : i32, 3 : i32]
  // CHECK-SAME: : (tensor<2x1x3x1xbf16>) -> tensor<1x1x2x3xbf16>
  // CHECK: return %[[R2]] : tensor<1x1x2x3xbf16>

  // Rank-5: singleton dims move, non-singletons keep order.
  func.func @permute_rank5_singletons(%arg0: tensor<1x4x1x3x1xbf16>)
      -> tensor<1x1x4x3x1xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 0, 2, 1, 3, 4> }
         : (tensor<1x4x1x3x1xbf16>) -> tensor<1x1x4x3x1xbf16>
    func.return %0 : tensor<1x1x4x3x1xbf16>
  }

  // CHECK-LABEL: @permute_rank5_singletons(
  // CHECK: %[[R3:.*]] = "ttnn.reshape"(%arg0)
  // CHECK-SAME: shape = [1 : i32, 1 : i32, 4 : i32, 3 : i32, 1 : i32]
  // CHECK-SAME: : (tensor<1x4x1x3x1xbf16>) -> tensor<1x1x4x3x1xbf16>
  // CHECK: return %[[R3]] : tensor<1x1x4x3x1xbf16>

  // Negative: swaps non-singleton dims, should stay permute.
  func.func @permute_reorder_non_singletons(%arg0: tensor<4x3x1x5xbf16>)
      -> tensor<3x4x1x5xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 1, 0, 2, 3> }
         : (tensor<4x3x1x5xbf16>) -> tensor<3x4x1x5xbf16>
    func.return %0 : tensor<3x4x1x5xbf16>
  }

  // CHECK-LABEL: @permute_reorder_non_singletons(
  // CHECK: %[[P0:.*]] = "ttnn.permute"(%arg0)
  // CHECK-SAME: permutation = array<i64: 1, 0, 2, 3>
  // CHECK-NOT: "ttnn.reshape"
  // CHECK: return %[[P0]] : tensor<3x4x1x5xbf16>

  // Negative: generic rank-3 permute, should not be rewritten.
  func.func @permute_rank3_keep(%arg0: tensor<4x3x5xbf16>)
      -> tensor<4x5x3xbf16> {
    %0 = "ttnn.permute"(%arg0)
         { permutation = array<i64: 0, 2, 1> }
         : (tensor<4x3x5xbf16>) -> tensor<4x5x3xbf16>
    func.return %0 : tensor<4x5x3xbf16>
  }

  // CHECK-LABEL: @permute_rank3_keep(
  // CHECK: %[[P1:.*]] = "ttnn.permute"(%arg0)
  // CHECK-SAME: permutation = array<i64: 0, 2, 1>
  // CHECK-NOT: "ttnn.reshape"
  // CHECK: return %[[P1]] : tensor<4x5x3xbf16>
}
