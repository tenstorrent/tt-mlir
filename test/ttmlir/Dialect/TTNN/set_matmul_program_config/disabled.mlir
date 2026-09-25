// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole enable-matmul-program-config=false" -mlir-print-local-scope %s | FileCheck %s
// Test that enable-matmul-program-config=false leaves matmuls to tt-metal.

// CHECK-LABEL: func.func @matmul_disabled
func.func @matmul_disabled(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
