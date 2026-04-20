// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // 1D indices (N,) must be unsqueezed to (1, N) before ttnn.embedding_bw,
  // otherwise tt-metal reshapes them to (N, 1, 1, N) and the internal
  // assertion N*N == N fails.
  func.func @backward_1d_indices(%arg0: tensor<32xsi32>, %arg1: tensor<512x128xbf16>, %arg2: tensor<32x128xbf16>) -> tensor<512x128xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 32 : i32]}>
    // CHECK-SAME: tensor<32xsi32
    // CHECK: "ttnn.embedding_bw"
    %1 = "ttir.embedding_backward"(%arg0, %arg1, %arg2) : (tensor<32xsi32>, tensor<512x128xbf16>, tensor<32x128xbf16>) -> tensor<512x128xbf16>
    return %1 : tensor<512x128xbf16>
  }
}
