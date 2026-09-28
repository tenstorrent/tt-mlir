// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 mock-system-desc-arch=blackhole compute-cfg-math-fidelity=hifi4 compute-cfg-fp32-dest-acc-en=true tensor-l1-usage-cap=0.8" -mlir-print-local-scope %s | FileCheck %s
// Test that with the optimizer enabled a config failing OpModel validation is
// dropped. The pass budgets 1.4 MB of circular buffers, but the 0.8 tensor L1
// usage cap leaves OpModel about 1.18 MB, below the 2D block the pass picks.

// CHECK-LABEL: func.func @matmul_2d
func.func @matmul_2d(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
