// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a matmul with transpose_a is left to tt-metal.

// CHECK-LABEL: func.func @matmul_transpose_a
func.func @matmul_transpose_a(%arg0: tensor<8192x1024xbf16>, %arg1: tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: transpose_a = true
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8192x1024xbf16>, tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}
