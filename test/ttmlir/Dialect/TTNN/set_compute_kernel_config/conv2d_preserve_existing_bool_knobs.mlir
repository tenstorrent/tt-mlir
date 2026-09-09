// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=unset math-approx-mode=false packer-l1-acc=true" %s | FileCheck %s
// Test tri-state handling of the new bool knobs:
//   - packer_l1_acc is already set to false on the op, so the pass must NOT
//     override it to true (existing op value wins).
//   - math_approx_mode is unset on the op, so the pass applies its value (false).
//   - fp32_dest_acc_en pipeline option is unset, so it is left untouched.
//   - math_fidelity is undefined, so it is left untouched.

// CHECK-LABEL: func @test_preserve_bool_knobs
func.func @test_preserve_bool_knobs(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %device: !ttnn.device) -> tensor<1x1x900x64xbf16> {
  // CHECK: ttnn.conv2d
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_approx_mode = false, packer_l1_acc = false>
  // CHECK-NOT: packer_l1_acc = true
  // CHECK-NOT: fp32_dest_acc_en
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
    groups = 1: i32,
    compute_config = #ttnn.device_compute_kernel_config<packer_l1_acc = false>
  } : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>

  return %result : tensor<1x1x900x64xbf16>
}
