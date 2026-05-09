// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify TTIR -> TTNN lowering of ttir.slice_reshape forwards begins / ends
// / step / shape onto ttnn.slice_reshape. Runtime composes
// ::ttnn::slice + ::ttnn::reshape in a single program-executor case.

module {
  // Quetzal pattern: slice the last dim of a fused QKV matmul output
  // (here 1x1x6144 -> 1x1x2048 is the Q chunk), then reshape into
  // (batch, heads, head_dim). Runtime calls ttnn::slice + ttnn::reshape
  // in a single program-executor case.
  // CHECK-LABEL: func.func @forward_slice_reshape
  // CHECK: "ttnn.slice_reshape"
  // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
  // CHECK-SAME: ends = [1 : i32, 1 : i32, 2048 : i32]
  // CHECK-SAME: shape = [1 : i32, 32 : i32, 64 : i32]
  // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32]
  func.func @forward_slice_reshape(%arg0: tensor<1x1x6144xbf16>) -> tensor<1x32x64xbf16> {
    %0 = "ttir.slice_reshape"(%arg0) <{
      begins = [0 : i32, 0 : i32, 0 : i32],
      ends = [1 : i32, 1 : i32, 2048 : i32],
      step = [1 : i32, 1 : i32, 1 : i32],
      shape = [1 : i32, 32 : i32, 64 : i32]
    }> : (tensor<1x1x6144xbf16>) -> tensor<1x32x64xbf16>
    return %0 : tensor<1x32x64xbf16>
  }
}
