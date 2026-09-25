// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole compute-cfg-math-fidelity=undefined compute-cfg-fp32-dest-acc-en=unset" -mlir-print-local-scope %s | FileCheck %s
// Test that a matmul without a compute_config is left to tt-metal: once a
// program config is present tt-metal would lower its default math fidelity.

// CHECK-LABEL: func.func @matmul_no_compute_config
func.func @matmul_no_compute_config(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: compute_config
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
