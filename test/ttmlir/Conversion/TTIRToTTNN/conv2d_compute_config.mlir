// RUN: ttmlir-opt --ttcore-register-device --ttir-flatten-sliding-window --ttnn-layout --convert-ttir-to-ttnn -o %t %s
// RUN: FileCheck %s --input-file=%t

// Quetzal pilot 1.4 / 3.3: per-op lowering must emit a HiFi4 +
// fp32_dest_acc_en=true compute_config on every ttnn.conv2d. Same rationale
// as matmul_compute_config.mlir.

module {
  // CHECK-LABEL: func.func @conv2d_default_compute_config
  func.func @conv2d_default_compute_config(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<
    // CHECK-SAME: math_fidelity = hifi4
    // CHECK-SAME: fp32_dest_acc_en = true
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1 : i32,
              padding = 0 : i32,
              dilation = 1 : i32,
              groups = 1 : i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %0 : tensor<1x30x30x64xbf16>
  }
}
