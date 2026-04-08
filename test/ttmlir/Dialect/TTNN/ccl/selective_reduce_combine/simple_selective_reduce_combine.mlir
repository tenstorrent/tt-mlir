// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn selective_reduce_combine op

// Verify lowering of ttir selective_reduce_combine to ttnn ops

module attributes {} {
  // CHECK-LABEL: selective_reduce_combine_basic
  func.func @selective_reduce_combine_basic(%arg0: tensor<4x2x128x2880xbf16>, %arg1: tensor<4x2x128x2880xbf16>, %arg2: tensor<1x2x128x4xi64>, %arg3: tensor<1x2x128x1xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttir.selective_reduce_combine"(%arg0, %arg1, %arg2, %arg3) <{hidden_size = 2880 : ui32, batch_size = 1 : ui32, seq_size = 128 : ui32, select_experts_k = 4 : ui32, experts = 32 : ui32, axis = 0 : ui32}> : (tensor<4x2x128x2880xbf16>, tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x2x128x1xi64>) -> tensor<4x1x128x2880xbf16>
    // CHECK: "ttnn.selective_reduce_combine"
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
