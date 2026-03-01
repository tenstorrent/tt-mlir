// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for reduce_scatter compute config workaround

// Verify that reduce_scatter gets a high-precision compute config with
// fp32_dest_acc_en=true when no config is present.

module attributes {} {
  // CHECK-LABEL: reduce_scatter_fp32_accum_config
  func.func @reduce_scatter_fp32_accum_config(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = "ttir.reduce_scatter"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>
    return %0 : tensor<1x1x8192x256xf32>
  }
}

// -----

// Verify that the workaround also applies to bf16 inputs.

module attributes {} {
  // CHECK-LABEL: reduce_scatter_fp32_accum_config_bf16
  func.func @reduce_scatter_fp32_accum_config_bf16(%arg0: tensor<1x1x8192x256xbf16>) -> tensor<1x1x8192x256xbf16> {
    %0 = "ttir.reduce_scatter"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xbf16>) -> tensor<1x1x8192x256xbf16>
    // CHECK: "ttnn.reduce_scatter"
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = false>
    return %0 : tensor<1x1x8192x256xbf16>
  }
}
