// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=false" %s | FileCheck %s

// CHECK-LABEL: func @test_large_k_linear
func.func @test_large_k_linear(%arg0: tensor<1x50001xbf16>, %arg1: tensor<128x50001xbf16>, %arg2: tensor<128xbf16>) -> tensor<1x128xbf16> {
  // CHECK: "ttnn.linear"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = true, packer_l1_acc = true>
  %0 = "ttnn.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1x50001xbf16>, tensor<128x50001xbf16>, tensor<128xbf16>) -> tensor<1x128xbf16>
  return %0 : tensor<1x128xbf16>
}
