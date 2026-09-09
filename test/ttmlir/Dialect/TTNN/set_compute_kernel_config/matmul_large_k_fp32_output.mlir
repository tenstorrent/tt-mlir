// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=false" %s | FileCheck %s

// CHECK-LABEL: func @test_large_k_fp32_matmul
func.func @test_large_k_fp32_matmul(%arg0: tensor<1x50001xf32>, %arg1: tensor<50001x128xf32>) -> tensor<1x128xf32> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = true>
  // CHECK-NOT: packer_l1_acc = true
  %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> {
    compute_config = #ttnn.device_compute_kernel_config<fp32_dest_acc_en = true>
  } : (tensor<1x50001xf32>, tensor<50001x128xf32>) -> tensor<1x128xf32>
  return %0 : tensor<1x128xf32>
}
