// Tri-state test for a single bool compute-kernel-config knob (packer_l1_acc):
// the pipeline option can force it ON (true), force it OFF (false), or be left
// UNSET (leave the op untouched so TTNN decides). math_fidelity and
// fp32_dest_acc_en are pinned to undefined/unset here so only packer_l1_acc is
// under test.

// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=unset packer-l1-acc=true" %s | FileCheck %s --check-prefix=TRUE
// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=unset packer-l1-acc=false" %s | FileCheck %s --check-prefix=FALSE
// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=undefined fp32-dest-acc-en=unset packer-l1-acc=unset" %s | FileCheck %s --check-prefix=UNSET

// TRUE-LABEL: func @test_packer_l1_acc_tristate
// FALSE-LABEL: func @test_packer_l1_acc_tristate
// UNSET-LABEL: func @test_packer_l1_acc_tristate
func.func @test_packer_l1_acc_tristate(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %device: !ttnn.device) -> tensor<1x1x900x64xbf16> {
  // TRUE: ttnn.conv2d
  // TRUE-SAME: compute_config = #ttnn.device_compute_kernel_config<packer_l1_acc = true>

  // FALSE: ttnn.conv2d
  // FALSE-SAME: compute_config = #ttnn.device_compute_kernel_config<packer_l1_acc = false>

  // UNSET: ttnn.conv2d
  // UNSET-NOT: compute_config
  // UNSET-NOT: device_compute_kernel_config
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
