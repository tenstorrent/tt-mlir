// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=lofi fp32-dest-acc-en=true" %s | FileCheck %s
// Test that the pass can set multiple parameters at once

// CHECK-LABEL: func @test_set_multiple_params
func.func @test_set_multiple_params(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %device: !ttnn.device) -> tensor<1x1x900x64xbf16> {
  // CHECK: ttnn.conv2d
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi, fp32_dest_acc_en = true>
  %result = "ttnn.conv2d"(%arg0, %arg1, %device) {
    in_channels = 64: i32,
    out_channels = 64: i32,
    batch_size = 1: i32,
    input_height = 32: i32,
    input_width = 32: i32,
    kernel_size = array<i32: 3, 3>,
    stride = array<i32: 1, 1>,
    padding = array<i32: 0, 0>,
    dilation = array<i32: 1, 1>,
    groups = 1: i32
  } : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>

  return %result : tensor<1x1x900x64xbf16>
}
