// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a batched B broadcast over an unbatched A is left to tt-metal.

// CHECK-LABEL: func.func @matmul_batch_broadcast
func.func @matmul_batch_broadcast(%arg0: tensor<1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
  return %0 : tensor<8x1024x1024xbf16>
}
