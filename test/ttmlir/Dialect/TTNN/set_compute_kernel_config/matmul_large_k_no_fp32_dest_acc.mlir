// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=false" %s | FileCheck %s

// CHECK-LABEL: func @test_large_k_bf16_matmul_no_config
func.func @test_large_k_bf16_matmul_no_config(%arg0: tensor<1x50001xbf16>, %arg1: tensor<50001x128xbf16>) -> tensor<1x128xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = true, packer_l1_acc = true>
  %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x50001xbf16>, tensor<50001x128xbf16>) -> tensor<1x128xbf16>
  return %0 : tensor<1x128xbf16>
}
